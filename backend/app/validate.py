"""Citation validation (answer pipeline step 5). Pure functions, no I/O.

The generator tags every factual sentence with excerpt aliases ([C1], [C2]...). Here:
  1. tags are mapped back to chunk_ids; tags that were not in the context are dropped;
  2. a sentence (or scheme field) with no valid citation is dropped;
  3. every number in a kept sentence (amounts, ages, dates; Malayalam digits, ₹/Rs,
     Indian digit grouping and lakh/crore normalised) must appear in a cited chunk or
     in the user's own question (their age, income...), otherwise the sentence is
     dropped;
  4. scheme cards lose fields that fail the same checks (kept fields carry their own
     [chunk_id] tags; an untagged field is checked against, and tagged with, the card's
     citations); eligible=true/false without a surviving reason becomes null; a scheme
     with no valid citation is dropped.
If no answer sentence survives, the caller abstains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

ML_DIGITS = str.maketrans("൦൧൨൩൪൫൬൭൮൯", "0123456789")
TAG = re.compile(r"\[([^\[\]\n]{1,120})\]")
ALIAS = re.compile(r"^C\d{1,2}$", re.I)
_UNIT = r"lakhs?|lacs?|crores?|thousand|ലക്ഷം|ലക്ഷ|കോടി|ആയിരം"
NUMBER = re.compile(rf"(?<![\w.])(\d+(?:\.\d+)?)(?:\s*({_UNIT}))?", re.I)
MULTIPLIER = {"lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
              "ലക്ഷം": 100_000, "ലക്ഷ": 100_000, "crore": 10_000_000, "crores": 10_000_000,
              "കോടി": 10_000_000, "thousand": 1000, "ആയിരം": 1000}  # fmt: skip
# Small numbers written as words in the source ("three instalments", "മൂന്ന് ഗഡു").
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "hundred": 100,
    "ഒന്ന": 1, "ഒരു": 1, "രണ്ട": 2, "മൂന്ന": 3, "നാല": 4, "അഞ്ച": 5, "ആറ": 6,
    "ഏഴ": 7, "എട്ട": 8, "ഒമ്പത": 9, "ഒൻപത": 9, "പത്ത": 10, "ഇരുപത": 20, "മുപ്പത": 30,
    "നാല്പത": 40, "അമ്പത": 50, "അറുപത": 60, "നൂറ": 100,
}  # fmt: skip
_WORD = re.compile(r"[A-Za-z]+|[ഀ-ൿ]+")
# Abbreviation dots that must not end a sentence.
_ABBREV = re.compile(r"\b(Rs|No|Nos|Dr|Smt|Sri|Shri|Govt|Dept|e\.g|i\.e|etc|viz|approx)\.",
                     re.I)  # fmt: skip
_DOT = ""
_SENT_END = re.compile(r"(?<=[.!?।])((?:\s*\[[^\[\]\n]{1,120}\])*)(?=\s|$)")


# ------------------------------------------------------------------ numbers


def normalise_numbers(text: str) -> str:
    """Malayalam digits -> ASCII, drop currency marks and digit-group commas/spaces."""
    t = text.translate(ML_DIGITS)
    t = re.sub(r"[    ]", " ", t)
    t = re.sub(r"₹|\bRs\.?|\bINR\b|രൂ(?:പ)?", " ", t, flags=re.I)
    t = re.sub(r"(?<=\d)[,٬](?=\d)", "", t)  # 1,20,000 -> 120000
    return re.sub(r"(?<=\d)/-", "", t)  # "6000/-"


def _canon(value: Decimal) -> str:
    v = value.normalize()
    return format(v, "f") if v == v.to_integral() else str(v)


def numbers_in(text: str, *, words: bool = False) -> set[str]:
    """Canonical numbers in ``text``; ``1.5 lakh`` yields both "1.5" and "150000"."""
    out: set[str] = set()
    for m in NUMBER.finditer(normalise_numbers(TAG.sub(" ", text))):
        try:
            value = Decimal(m.group(1))
        except InvalidOperation:
            continue
        out.add(_canon(value))
        if m.group(2):
            out.add(_canon(value * MULTIPLIER[m.group(2).lower()]))
    if words:
        for w in _WORD.findall(text.lower()):
            for stem, n in NUMBER_WORDS.items():
                if w == stem or (stem[0] > "z" and w.startswith(stem)):
                    out.add(str(n))
    return out


def unsupported_numbers(
    claim: str, sources: list[str], allowed: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    """Numbers in ``claim`` (outside tags) not in any of ``sources`` nor in ``allowed``."""
    have: set[str] = set(allowed)
    for s in sources:
        have |= numbers_in(s, words=True)
    missing = []
    for m in NUMBER.finditer(normalise_numbers(TAG.sub(" ", claim))):
        try:
            value = Decimal(m.group(1))
        except InvalidOperation:
            continue
        forms = {_canon(value)}
        if m.group(2):
            forms.add(_canon(value * MULTIPLIER[m.group(2).lower()]))
        if not forms & have:
            missing.append(m.group(0).strip())
    return missing


# ------------------------------------------------------------------ sentences


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? । (keeping trailing citation tags with their sentence) and newlines."""
    protected = _ABBREV.sub(lambda m: m.group(0)[:-1] + _DOT, text)
    out: list[str] = []
    for line in protected.splitlines():
        start = 0
        for m in _SENT_END.finditer(line):
            out.append(line[start : m.end()])
            start = m.end()
        out.append(line[start:])
    return [s.replace(_DOT, ".").strip() for s in out if s.strip()]


def tags_in(text: str) -> list[str]:
    """Raw citation labels in order: "[C1, C3]" and "[C1][C3]" both give C1, C3."""
    labels: list[str] = []
    for m in TAG.finditer(text):
        for part in re.split(r"[,;\s]+", m.group(1).strip()):
            if part and part not in labels:
                labels.append(part)
    return labels


def strip_tags(text: str) -> str:
    return re.sub(r"\s+([.!?।,])", r"\1", re.sub(r"\s*" + TAG.pattern, "", text)).strip()


def is_citation_like(label: str) -> bool:
    return bool(ALIAS.match(label)) or ":" in label


# ------------------------------------------------------------------ validation


@dataclass
class Dropped:
    text: str
    reason: str  # no_citation | unknown_citation | number_mismatch | entailment
    detail: str = ""


@dataclass
class Validation:
    answer: str  # kept sentences, tags rewritten to [chunk_id]
    sentences: list[dict]  # [{"text", "citations": [chunk_id]}] (kept)
    schemes: list[dict]
    cited_ids: list[str]
    dropped: list[Dropped] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(not s["text"].rstrip().endswith("?") for s in self.sentences)


class Validator:
    """Validates one generation against the excerpts it was given.

    ``aliases`` maps "C1" -> chunk_id; ``chunks`` maps chunk_id -> chunk dict (with
    "text"); ``user_text`` is the question plus stated profile facts.
    """

    def __init__(
        self, aliases: dict[str, str], chunks: dict[str, dict], user_text: str = ""
    ) -> None:
        self.aliases = {k.upper(): v for k, v in aliases.items()}
        self.chunks = chunks
        self.user_numbers = numbers_in(user_text)  # facts the user stated may be repeated
        self.unknown: list[str] = []
        self.dropped: list[Dropped] = []

    def resolve(self, labels: list[str]) -> list[str]:
        ids: list[str] = []
        for label in labels:
            cid = self.aliases.get(label.upper()) or (label if label in self.chunks else None)
            if cid is None:
                if is_citation_like(label) and label not in self.unknown:
                    self.unknown.append(label)
                continue
            if cid not in ids:
                ids.append(cid)
        return ids

    def check(self, text: str, fallback: list[str] | None = None) -> tuple[str, list[str]] | None:
        """Return (clean text, chunk ids) if ``text`` is supported, else None (dropped)."""
        labels = [lab for lab in tags_in(text) if is_citation_like(lab)]
        ids = self.resolve(labels)
        if not ids:
            if labels:
                self.dropped.append(Dropped(text, "unknown_citation", ", ".join(labels)))
                return None
            if not fallback:
                self.dropped.append(Dropped(text, "no_citation"))
                return None
            ids = list(fallback)
        bad = unsupported_numbers(text, [self.chunks[c]["text"] for c in ids], self.user_numbers)
        if bad:
            self.dropped.append(Dropped(text, "number_mismatch", ", ".join(bad)))
            return None
        return strip_tags(text), ids

    def validate(self, data: dict, scheme_ids: dict[str, str]) -> Validation:
        """``data`` is the parsed generator JSON; ``scheme_ids`` maps chunk_id -> scheme_id."""
        flags: list[str] = []
        kept: list[dict] = []
        for sent in split_sentences(str(data.get("answer") or "")):
            if sent.rstrip().endswith("?") and not tags_in(sent):
                kept.append({"text": sent, "citations": []})  # a question to the user
                continue
            res = self.check(sent)
            if res:
                kept.append({"text": res[0], "citations": res[1]})
        schemes = [s for s in (self._scheme(raw, scheme_ids, flags)
                               for raw in data.get("schemes") or [] if isinstance(raw, dict))
                   if s]  # fmt: skip
        cited: list[str] = []
        for cid in [c for s in kept for c in s["citations"]] + [
            c for s in schemes for c in s["citations"]
        ]:
            if cid not in cited:
                cited.append(cid)
        answer = " ".join(
            s["text"] + "".join(f" [{c}]" for c in s["citations"]) if s["citations"] else s["text"]
            for s in kept
        )
        return Validation(answer, kept, schemes, cited, self.dropped, self.unknown, flags)

    def _field(self, value: object, fallback: list[str], used: list[str]) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        res = self.check(value, fallback)
        if not res:
            return None
        used.extend(c for c in res[1] if c not in used)
        return res[0] + "".join(f" [{c}]" for c in res[1])  # each field keeps its own tags

    def _scheme(self, raw: dict, scheme_ids: dict[str, str], flags: list[str]) -> dict | None:
        labels = [str(c) for c in raw.get("citations") or [] if isinstance(c, str | int)]
        base = self.resolve([lab.strip("[] ") for lab in labels])
        sid = str(raw.get("scheme_id") or "")
        context_schemes = {scheme_ids[c] for c in self.aliases.values() if c in scheme_ids}
        if sid not in context_schemes:
            # the model may have used a name instead of the id: take it from the citations
            from_cites = {scheme_ids.get(c) for c in base} - {None}
            if len(from_cites) == 1:
                sid = from_cites.pop()
            else:
                flags.append(f"scheme_dropped_unknown_id:{sid or raw.get('name')}")
                return None
        # field checks may fall back to this scheme's own chunks only
        base = [c for c in base if scheme_ids.get(c) == sid]
        used: list[str] = []
        reasons = [r for r in (self._field(x, base, used) for x in raw.get("reasons") or []) if r]
        documents = [d for d in (self._field(x, base, used) for x in raw.get("documents") or [])
                     if d]  # fmt: skip
        benefit = self._field(raw.get("benefit"), base, used)
        how = self._field(raw.get("how_to_apply"), base, used)
        citations = list(dict.fromkeys(base + used))
        if not citations:
            flags.append(f"scheme_dropped_no_citation:{sid}")
            return None
        eligible = raw.get("eligible")
        if eligible not in (True, False, None):
            eligible = None
        if eligible is not None and not reasons:
            flags.append(f"eligible_nulled_no_reason:{sid}")
            eligible = None
        return {"scheme_id": sid, "name": str(raw.get("name") or sid), "eligible": eligible,
                "reasons": reasons, "benefit": benefit, "how_to_apply": how,
                "documents": documents, "citations": citations}  # fmt: skip


# ------------------------------------------------------------------ card filter

_RULE_SECTIONS = {"eligibility", "exclusions", "benefits"}
_OTHER_SECTIONS = {"how_to_apply", "documents", "objectives"}
_RULE_WORDS = re.compile(
    r"eligib|benefit|assistance|income|age of|years of age|entitled|pension of|"
    r"അർഹ|യോഗ്യ|ആനുകൂല്യ|ധനസഹായ|വരുമാന|മാനദണ്ഡ|വയസ്|പെൻഷൻ തുക|സഹായം",
    re.I,
)
TOP_SCHEMES = 3


def is_rule_chunk(chunk: dict) -> bool:
    """An eligibility or benefit chunk: by section label, else by its wording (PDFs have none)."""
    section = chunk.get("section")
    if section in _RULE_SECTIONS:
        return True
    return section not in _OTHER_SECTIONS and bool(_RULE_WORDS.search(chunk["text"]))


def filter_cards(
    schemes: list[dict],
    answer_cited: list[str],
    ranked_ids: list[str],
    chunks: dict[str, dict],
    scheme_ids: dict[str, str],
) -> tuple[list[dict], list[str]]:
    """Drop scheme cards the answer does not rest on. Returns (kept cards, dropped scheme ids).

    A card needs an eligibility/benefit chunk of its scheme among the answer's and the card's
    citations. It must also be a scheme the answer cites; when the answer cites nothing (it is
    about to be rebuilt from the cards), a scheme among the top reranked ones instead.
    """
    answer_schemes = {scheme_ids.get(c) for c in answer_cited}
    top: list[str] = []
    for cid in ranked_ids:
        sid = scheme_ids.get(cid)
        if sid and sid not in top:
            top.append(sid)
    allowed = answer_schemes if answer_cited else set(top[:TOP_SCHEMES])
    kept, dropped = [], []
    for card in schemes:
        sid = card["scheme_id"]
        cited = [c for c in [*answer_cited, *card["citations"]] if scheme_ids.get(c) == sid]
        if sid in allowed and any(is_rule_chunk(chunks[c]) for c in cited):
            kept.append(card)
        else:
            dropped.append(sid)
    return kept, dropped
