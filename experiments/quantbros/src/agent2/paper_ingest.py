"""Paper extraction, heuristic classification, SQLite ingest, and fixture export."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from agent2.database import connect_sqlite, init_db, insert_extraction, upsert_paper, upsert_sections
from agent2.openrouter import OpenRouterClient, parse_json_text
from agent2.schemas import (
    SCHEMA_VERSION,
    Agent1Sections,
    Agent1ToAgent2Input,
    AssetUniverseType,
    OpenRouterCallMetadata,
    PaperCategory,
    PaperClassification,
    RegimeClaim,
)


LOGGER = logging.getLogger(__name__)
COMMON_TICKERS = {
    "DBC",
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "IEF",
    "GLD",
    "SLV",
    "EFA",
    "EEM",
    "VNQ",
    "USO",
    "UUP",
    "VXX",
    "BTC",
    "ETH",
}
INDEX_PROXY_PATTERNS = {
    r"S&P 500": "SPY",
    r"Dow Jones": "DIA",
    r"NASDAQ[- ]100": "QQQ",
    r"Russell 2000": "IWM",
    r"MSCI EAFE": "EFA",
    r"long term government bonds?": "TLT",
    r"short term government bills?": "IEF",
    r"REITs?": "VNQ",
    r"commodit(?:y|ies)": "DBC",
    r"gold": "GLD",
}
SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "methodology": "methodology",
    "methods": "methodology",
    "strategy": "methodology",
    "trading strategy": "methodology",
    "model specification": "methodology",
    "method": "methodology",
    "conclusion": "conclusion",
    "discussion": "conclusion",
}
STOPWORDS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "ETF",
    "FOR",
    "GDP",
    "IN",
    "OF",
    "ON",
    "OR",
    "PDF",
    "THE",
    "TO",
    "US",
    "USA",
    "USD",
}


@dataclass
class IngestSummary:
    ingested_count: int
    paper_ids: list[str]
    db_path: Path
    fixtures_dir: Path


def slugify_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "paper"


def iter_paper_sources(papers_dir: Path) -> Iterable[Path]:
    seen_txt_sidecars: set[Path] = set()
    pdfs = sorted(papers_dir.rglob("*.pdf"))
    for pdf in pdfs:
        seen_txt_sidecars.add(pdf.with_suffix(".txt"))
        yield pdf
    for txt in sorted(papers_dir.rglob("*.txt")):
        if txt not in seen_txt_sidecars:
            yield txt


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_garbled_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 300:
        return True
    alpha_ratio = sum(ch.isalpha() for ch in stripped) / max(len(stripped), 1)
    return alpha_ratio < 0.45


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return normalize_text("\n".join(parts))


def extract_pdf_text_pdftotext(path: Path) -> str:
    if shutil.which("pdftotext") is None:
        return ""
    result = subprocess.run(
        ["pdftotext", "-layout", "-nopgbrk", str(path), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    return normalize_text(result.stdout)


def extract_pdf_text_ocr(path: Path, *, max_pages: int = 8, dpi: int = 150) -> str:
    if shutil.which("pdftoppm") is None or shutil.which("tesseract") is None:
        return ""
    parts: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "1",
                "-l",
                str(max_pages),
                "-r",
                str(dpi),
                "-png",
                str(path),
                str(prefix),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for image_path in sorted(Path(tmpdir).glob("*.png")):
            result = subprocess.run(
                ["tesseract", str(image_path), "stdout", "--psm", "6"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                parts.append(result.stdout)
    return normalize_text("\n".join(parts))


def extract_raw_text(path: Path) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if path.suffix.lower() == ".pdf":
        text = extract_pdf_text(path)
        if is_garbled_text(text):
            sidecar = path.with_suffix(".txt")
            if sidecar.exists():
                warnings.append(f"pdf extraction looked weak; used sidecar {sidecar.name}")
                LOGGER.warning("Weak PDF extraction for %s; falling back to %s", path, sidecar)
                text = normalize_text(sidecar.read_text(encoding="utf-8", errors="ignore"))
        if is_garbled_text(text):
            fallback = extract_pdf_text_pdftotext(path)
            if fallback and not is_garbled_text(fallback):
                warnings.append("used pdftotext fallback for PDF extraction")
                text = fallback
        if is_garbled_text(text):
            fallback = extract_pdf_text_ocr(path)
            if fallback and not is_garbled_text(fallback):
                warnings.append("used OCR fallback for PDF extraction")
                LOGGER.warning("Used OCR fallback for %s", path)
                text = fallback
        return text, warnings
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore")), warnings


def infer_title(raw_text: str, source_path: Path) -> str:
    for line in raw_text.splitlines()[:20]:
        candidate = " ".join(line.split()).strip()
        if not candidate:
            continue
        if len(candidate) < 6 or len(candidate) > 180:
            continue
        if re.fullmatch(r"\d+", candidate):
            continue
        return candidate
    return source_path.stem.replace("_", " ").replace("-", " ").title()


def _find_heading_positions(lines: list[str]) -> list[tuple[int, str]]:
    positions: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        normalized = re.sub(r"[^a-z ]+", "", line.strip().lower())
        if normalized in SECTION_ALIASES:
            positions.append((idx, SECTION_ALIASES[normalized]))
    return positions


def extract_sections(raw_text: str) -> Agent1Sections:
    lines = [line.strip() for line in raw_text.splitlines()]
    headings = _find_heading_positions(lines)
    content: dict[str, str] = {}
    for pos, name in headings:
        next_positions = [idx for idx, _ in headings if idx > pos]
        end = next_positions[0] if next_positions else len(lines)
        body = "\n".join(line for line in lines[pos + 1 : end] if line).strip()
        if body and name not in content:
            content[name] = body

    abstract = content.get("abstract", raw_text[:1400]).strip()
    introduction = content.get("introduction", "")
    methodology = content.get("methodology", "")
    conclusion = content.get("conclusion", "")

    if not methodology:
        method_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", raw_text)
            if re.search(
                r"\b(rank|rebalance|hold|moving average|portfolio|buy|sell|return|signal|window|momentum)\b",
                sentence,
                flags=re.IGNORECASE,
            )
        ]
        methodology = " ".join(method_sentences[:8]).strip()
    notes = ""
    if not conclusion:
        notes = "Conclusion section not confidently extracted."

    return Agent1Sections(
        abstract=abstract[:5000],
        methodology=methodology[:6000],
        introduction=introduction[:5000],
        conclusion=conclusion[:4000],
        notes=notes,
    )


def extract_tickers(raw_text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for pattern, proxy in INDEX_PROXY_PATTERNS.items():
        if re.search(pattern, raw_text, flags=re.IGNORECASE) and proxy not in seen:
            ordered.append(proxy)
            seen.add(proxy)

    paren_tokens = re.findall(r"\(([A-Z]{2,5})\)", raw_text)
    tokens = paren_tokens + re.findall(r"\b[A-Z]{2,5}\b", raw_text)
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token in COMMON_TICKERS and token not in seen:
            ordered.append(token)
            seen.add(token)
        if len(ordered) >= 12:
            break
    return ordered


def _keyword_score(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(keyword) for keyword in keywords)


def heuristic_classification(
    *,
    title: str,
    sections: Agent1Sections,
    extracted_tickers: list[str],
) -> PaperClassification:
    combined = f"{title}\n{sections.abstract}\n{sections.methodology}".lower()

    category_scores = {
        PaperCategory.trend: _keyword_score(
            combined,
            ["moving average", "trend following", "trend", "timing", "tactical"],
        ),
        PaperCategory.momentum: _keyword_score(
            combined,
            ["momentum", "relative strength", "rank", "cross-sectional"],
        ),
        PaperCategory.mean_reversion: _keyword_score(
            combined,
            ["mean reversion", "reversal", "overreaction", "contrarian"],
        ),
        PaperCategory.value: _keyword_score(combined, ["value", "valuation", "book-to-market"]),
        PaperCategory.carry: _keyword_score(combined, ["carry", "roll yield"]),
        PaperCategory.volatility: _keyword_score(combined, ["volatility", "vix", "variance"]),
    }
    best_category = max(category_scores, key=category_scores.get)
    if category_scores[best_category] <= 0:
        best_category = PaperCategory.unknown

    if any(keyword in combined for keyword in ["multi-asset", "asset allocation", "stocks and bonds", "equity and bond"]):
        universe_type = AssetUniverseType.multi_asset
    elif any(keyword in combined for keyword in ["etf", "fund", "spy", "qqq", "iwm"]) or extracted_tickers:
        universe_type = AssetUniverseType.etfs
    elif any(keyword in combined for keyword in ["treasury", "bond", "fixed income"]):
        universe_type = AssetUniverseType.bonds
    elif any(keyword in combined for keyword in ["future", "futures"]):
        universe_type = AssetUniverseType.futures
    elif any(keyword in combined for keyword in ["fx", "currency", "foreign exchange"]):
        universe_type = AssetUniverseType.fx
    elif any(keyword in combined for keyword in ["crypto", "bitcoin", "ethereum"]):
        universe_type = AssetUniverseType.crypto
    elif any(keyword in combined for keyword in ["stock", "equity"]):
        universe_type = AssetUniverseType.equities
    else:
        universe_type = AssetUniverseType.unknown

    if any(keyword in combined for keyword in ["high volatility", "volatile", "crisis", "stress regime"]):
        regime_claim = RegimeClaim.high_volatility
    elif any(keyword in combined for keyword in ["low volatility", "calm market"]):
        regime_claim = RegimeClaim.low_volatility
    elif "bear market" in combined:
        regime_claim = RegimeClaim.bear
    elif "bull market" in combined:
        regime_claim = RegimeClaim.bull
    elif any(keyword in combined for keyword in ["across regimes", "different regimes", "all market conditions"]):
        regime_claim = RegimeClaim.mixed
    else:
        regime_claim = RegimeClaim.not_specified

    note_patterns = [
        r"S&P 500",
        r"Dow Jones",
        r"US large[- ]cap",
        r"industry portfolios",
        r"exchange-traded funds?",
        r"Treasur(?:y|ies)",
        r"commodit(?:y|ies)",
    ]
    notes = ""
    for pattern in note_patterns:
        match = re.search(pattern, title + "\n" + sections.abstract + "\n" + sections.methodology, flags=re.IGNORECASE)
        if match:
            notes = match.group(0)
            break
    if not notes and extracted_tickers:
        notes = f"Tickers mentioned: {', '.join(extracted_tickers[:6])}"

    populated_scores = [score for score in category_scores.values() if score > 0]
    confidence = min(0.95, 0.35 + 0.15 * len(populated_scores))

    return PaperClassification(
        paper_category=best_category,
        asset_universe_type=universe_type,
        asset_universe_notes=notes,
        regime_claim=regime_claim,
        confidence=confidence,
        method="heuristic",
        reasoning="Keyword-based classification from title, abstract, and methodology.",
    )


def _require_legacy_heuristics(allow_legacy_heuristics: bool) -> None:
    if not allow_legacy_heuristics:
        raise RuntimeError(
            "Legacy heuristic classification is disabled. Use agentic LLM classification or "
            "explicitly opt into the deprecated legacy path."
        )
    warnings.warn(
        "Using deprecated heuristic paper classification. Agentic LLM classification should be preferred.",
        DeprecationWarning,
        stacklevel=2,
    )


def llm_classification(
    *,
    title: str,
    sections: Agent1Sections,
) -> tuple[PaperClassification, OpenRouterCallMetadata]:
    client = OpenRouterClient()
    messages = [
        {
            "role": "system",
            "content": (
                "You classify quantitative finance papers into constrained enums. "
                "Return JSON only with keys paper_category, asset_universe_type, "
                "asset_universe_notes, regime_claim, confidence, reasoning."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "title": title,
                    "abstract": sections.abstract[:2500],
                    "methodology": sections.methodology[:3000],
                    "allowed_paper_category": [item.value for item in PaperCategory],
                    "allowed_asset_universe_type": [item.value for item in AssetUniverseType],
                    "allowed_regime_claim": [item.value for item in RegimeClaim],
                }
            ),
        },
    ]
    response = client.chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
    )
    payload = parse_json_text(response.content)
    model = PaperClassification.model_validate(payload).model_copy(update={"method": "openrouter"})
    metadata = OpenRouterCallMetadata(
        model=response.model,
        latency_seconds=response.latency_seconds,
        usage=response.usage,
    )
    return model, metadata


def classify_paper(
    *,
    title: str,
    sections: Agent1Sections,
    extracted_tickers: list[str],
    use_llm: bool,
    allow_legacy_heuristics: bool = False,
) -> tuple[PaperClassification, OpenRouterCallMetadata | None]:
    if use_llm:
        llm_model, metadata = llm_classification(title=title, sections=sections)
        return llm_model, metadata
    _require_legacy_heuristics(allow_legacy_heuristics)
    heuristic = heuristic_classification(
        title=title,
        sections=sections,
        extracted_tickers=extracted_tickers,
    )
    return heuristic, None


def infer_logic_hints(
    *,
    title: str,
    sections: Agent1Sections,
    extracted_tickers: list[str],
    classification: PaperClassification,
) -> dict[str, object]:
    text = f"{title}\n{sections.abstract}\n{sections.methodology}".lower()
    hints: dict[str, object] = {}
    if "moving average" in text or "crosses above" in text or classification.paper_category == PaperCategory.trend:
        windows = [int(value) for value in re.findall(r"(\d+)[-\s]day moving average", text)]
        if len(windows) >= 2:
            fast, slow = sorted(windows[:2])[:2]
        else:
            fast, slow = 20, 50
        hints.update(
            {
                "suggested_strategy": "dual_moving_average",
                "fast_window": fast,
                "slow_window": slow,
                "tickers": extracted_tickers[:1] or ["SPY"],
                "preset": "spy_dual_ma",
            }
        )
    elif "momentum" in text or "rank" in text or classification.paper_category == PaperCategory.momentum:
        top_match = re.search(r"top\s+(\d+)", text)
        top_n = int(top_match.group(1)) if top_match else 3
        if "12 month" in text or "twelve month" in text:
            lookback = 252
        elif "6 month" in text or "six month" in text:
            lookback = 126
        else:
            lookback = 126
        tickers = extracted_tickers[:6] or ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD"]
        hints.update(
            {
                "suggested_strategy": "momentum_rank",
                "lookback_days": lookback,
                "top_n": min(top_n, max(1, len(tickers))),
                "tickers": tickers,
                "preset": "etf_momentum_rank",
            }
        )
    return hints


def export_fixture(path: Path, payload: Agent1ToAgent2Input) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")


def ingest_papers(
    *,
    papers_dir: str | Path,
    db_path: str | Path,
    fixtures_dir: str | Path,
    use_llm_classifier: bool = True,
    allow_legacy_heuristics: bool = False,
) -> IngestSummary:
    papers_root = Path(papers_dir)
    fixtures_root = Path(fixtures_dir)
    conn = connect_sqlite(db_path)
    init_db(conn)

    ingested: list[str] = []
    for source in iter_paper_sources(papers_root):
        raw_text, warnings = extract_raw_text(source)
        if not raw_text:
            warnings.append("text extraction failed; ingested placeholder metadata only")
            LOGGER.warning("Text extraction failed for %s; ingesting placeholder row", source)

        paper_id = slugify_name(source.stem)
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        title = infer_title(raw_text, source)
        sections = extract_sections(raw_text)
        tickers = extract_tickers(raw_text)
        classification, metadata = classify_paper(
            title=title,
            sections=sections,
            extracted_tickers=tickers,
            use_llm=use_llm_classifier,
            allow_legacy_heuristics=allow_legacy_heuristics,
        )
        logic_hints = (
            infer_logic_hints(
                title=title,
                sections=sections,
                extracted_tickers=tickers,
                classification=classification,
            )
            if allow_legacy_heuristics
            else {}
        )
        parsed_at = datetime.now(timezone.utc).isoformat()
        raw_metadata: dict[str, object] = {
            "parsed_at": parsed_at,
            "source_filename": source.name,
            "content_hash": content_hash,
            "extracted_tickers": tickers,
            "ingest_warnings": warnings,
            "classification_confidence": classification.confidence,
            "classification_method": classification.method,
            "classification_reasoning": classification.reasoning,
            "logic_hints": logic_hints,
        }
        if metadata is not None:
            raw_metadata["classification_model"] = metadata.model
            raw_metadata["classification_usage"] = metadata.usage

        payload = Agent1ToAgent2Input(
            schema_version=SCHEMA_VERSION,
            paper_id=paper_id,
            title=title,
            source_path=str(source),
            parsed_at=parsed_at,
            sections=sections,
            extracted_tickers=tickers,
            paper_category=classification.paper_category,
            asset_universe_type=classification.asset_universe_type,
            asset_universe_notes=classification.asset_universe_notes,
            regime_claim=classification.regime_claim,
            raw_metadata=raw_metadata,
        )

        upsert_paper(
            conn,
            paper_id=paper_id,
            schema_version=SCHEMA_VERSION,
            title=title,
            source_path=str(source),
            created_at=parsed_at,
            content_hash=content_hash,
            raw_text=raw_text,
            paper_category=classification.paper_category,
            asset_universe_type=classification.asset_universe_type,
            asset_universe_notes=classification.asset_universe_notes,
            regime_claim=classification.regime_claim,
            metadata=raw_metadata,
        )
        upsert_sections(conn, paper_id, sections)
        if metadata is not None:
            insert_extraction(
                conn,
                paper_id=paper_id,
                extraction_type="paper_classification",
                extracted_at=parsed_at,
                model=metadata.model,
                confidence=classification.confidence,
                payload=classification.model_dump(mode="json"),
            )

        export_fixture(fixtures_root / f"{paper_id}.json", payload)
        ingested.append(paper_id)

    return IngestSummary(
        ingested_count=len(ingested),
        paper_ids=ingested,
        db_path=Path(db_path),
        fixtures_dir=fixtures_root,
    )
