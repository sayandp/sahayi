export type UiLang = "ml" | "en";

export const UI = {
  ml: {
    heading: "നിങ്ങൾക്ക് ഏത് സർക്കാർ പദ്ധതി ലഭിക്കും?",
    sub: "മലയാളത്തിലോ ഇംഗ്ലീഷിലോ ചോദിക്കൂ. ഓരോ ഉത്തരത്തിനും ഔദ്യോഗിക രേഖയും പേജും കാണിക്കും.",
    placeholder: "ഉദാ: വിധവാ പെൻഷന് എനിക്ക് അർഹതയുണ്ടോ?",
    ask: "ചോദിക്കുക",
    try: "ഇവ ചോദിച്ചു നോക്കൂ",
    sources: "കണ്ടെത്തിയ രേഖകൾ",
    finding: "ഔദ്യോഗിക രേഖകൾ തിരയുന്നു",
    writing: "ഉത്തരം എഴുതുന്നു",
    schemes: "പദ്ധതികൾ",
    yes: "അർഹതയുണ്ട്",
    no: "അർഹതയില്ല",
    unknown: "വിവരം വേണം",
    unchecked: "അർഹത പരിശോധിച്ചില്ല",
    why: "കാരണങ്ങൾ",
    benefit: "ലഭിക്കുന്നത്",
    how: "അപേക്ഷിക്കേണ്ട വിധം",
    docs: "വേണ്ട രേഖകൾ",
    followUp: "ഒരു കാര്യം കൂടി അറിയണം",
    yesReply: "അതെ",
    noReply: "അല്ല",
    replyPlaceholder: "നിങ്ങളുടെ മറുപടി",
    send: "അയയ്ക്കുക",
    noAnswer: "ഈ രേഖകളിൽ ഉത്തരമില്ല",
    busy: "ഇപ്പോൾ തിരക്കാണ്",
    verify: "അപേക്ഷിക്കുന്നതിന് മുമ്പ് അടുത്തുള്ള അക്ഷയ കേന്ദ്രത്തിൽ ഉറപ്പാക്കുക.",
    asOf: "വിവരങ്ങൾ ഈ തീയതി വരെ",
    newQuestion: "പുതിയ ചോദ്യം",
    source: "രേഖ",
    page: "പേജ്",
    openPdf: (n: number) => `PDF പേജ് ${n} തുറക്കുക`,
    openPage: "ഔദ്യോഗിക പേജ് തുറക്കുക",
    close: "അടയ്ക്കുക",
    settings: "ക്രമീകരണങ്ങൾ",
    theme: "നിറം",
    themes: { system: "ഫോൺ പോലെ", light: "വെളിച്ചം", dark: "ഇരുട്ട്" },
    speed: "അവസാന ഉത്തരത്തിന്റെ വേഗം",
    noSpeed: "ഒരു ചോദ്യം ചോദിച്ചാൽ ഇവിടെ കാണാം.",
    ttfc: "രേഖകൾ കാണിച്ചത്",
    ttft: "ഉത്തരം തുടങ്ങിയത്",
    total: "ആകെ",
    model: "മോഡൽ",
    cost: "ചെലവ്",
    fromCache: "സംഭരിച്ച ഉത്തരം",
    mock: "ഡെമോ: റെക്കോർഡ് ചെയ്ത ഉത്തരങ്ങൾ മാത്രം",
    mockMiss:
      "ഈ ഡെമോയിൽ ഉദാഹരണ ചോദ്യങ്ങൾക്ക് മാത്രമേ റെക്കോർഡ് ചെയ്ത ഉത്തരമുള്ളൂ. താഴെയുള്ളത് മറ്റൊരു ചോദ്യത്തിന്റെ ഉത്തരമാണ്.",
    errorTitle: "ഉത്തരം ലഭിച്ചില്ല",
    errorBody: "ഇന്റർനെറ്റ് പരിശോധിച്ച് വീണ്ടും ശ്രമിക്കുക.",
    rateLimited: "കുറേ ചോദ്യങ്ങൾ ഒന്നിച്ചായി. ഒരു മിനിറ്റ് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
    retry: "വീണ്ടും ശ്രമിക്കുക",
  },
  en: {
    heading: "Which government schemes can you get?",
    sub: "Ask in Malayalam or English. Every answer shows the official document and page it came from.",
    placeholder: "For example: Can I get the widow pension?",
    ask: "Ask",
    try: "Try asking",
    sources: "Documents found",
    finding: "Searching official documents",
    writing: "Writing the answer",
    schemes: "Schemes",
    yes: "Eligible",
    no: "Not eligible",
    unknown: "Need more details",
    unchecked: "Eligibility not checked",
    why: "Why",
    benefit: "What you get",
    how: "How to apply",
    docs: "Documents needed",
    followUp: "One more thing",
    yesReply: "Yes",
    noReply: "No",
    replyPlaceholder: "Your reply",
    send: "Send",
    noAnswer: "Not in these documents",
    busy: "Busy right now",
    verify: "Check with your nearest Akshaya centre before applying.",
    asOf: "Information as of",
    newQuestion: "New question",
    source: "Document",
    page: "Page",
    openPdf: (n: number) => `Open PDF at page ${n}`,
    openPage: "Open the official page",
    close: "Close",
    settings: "Settings",
    theme: "Colours",
    themes: { system: "Same as phone", light: "Light", dark: "Dark" },
    speed: "Speed of the last answer",
    noSpeed: "Ask a question to see this.",
    ttfc: "Documents shown",
    ttft: "Answer started",
    total: "Total",
    model: "Model",
    cost: "Cost",
    fromCache: "Saved answer",
    mock: "Demo: recorded answers only",
    mockMiss:
      "This demo only has recorded answers for the example questions. What follows is the answer to a different question.",
    errorTitle: "No answer came back",
    errorBody: "Check your internet connection and try again.",
    rateLimited: "Too many questions at once. Try again in a minute.",
    retry: "Try again",
  },
} as const;

export const EXAMPLES: { q: string; lang: "ml" | "en" }[] = [
  { q: "എന്റെ കുടുംബത്തിൽ NREGA ജോബ് കാർഡ് അപേക്ഷിക്കാൻ എന്തെല്ലാം രേഖകൾ വേണം?", lang: "ml" },
  { q: "നെൽകർഷകർക്ക് പെൻഷൻ കൊടുക്കാൻ തുടങ്ങിയത് എന്ന് മുതലാണ്?", lang: "ml" },
  {
    q: "I'm 45 years old and have a savings account. Can I still join Atal Pension Yojana?",
    lang: "en",
  },
  {
    q: "My husband passed away last year. Can I get the Indira Gandhi National Widow Pension?",
    lang: "en",
  },
  {
    q: "njan kerala resident aayi, disability 40% kooduthal ulla student aanu, family income 30k aanu, scholarship kittumo?",
    lang: "en",
  },
  { q: "karshaka thozhilali pensionu kshemanidhi board il ninnu enthu certificate venam?", lang: "en" },
];

/** Script of a text, for line-height and fonts: any Malayalam letter counts. */
export const langOf = (text: string): "ml" | "en" => (/[ഀ-ൿ]/.test(text) ? "ml" : "en");

const YES_NO_EN = /^(do|does|did|are|is|was|were|have|has|can|will|should)\b/i;
const YES_NO_ML = /(ആണോ|ഉണ്ടോ|ഇല്ലേ|അല്ലേ|കഴിയുമോ|ചെയ്യുന്നുണ്ടോ|ലഭിക്കുന്നുണ്ടോ)\s*\?$/;

/** Quick-reply buttons only fit a single yes/no question. */
export function isYesNo(question: string): boolean {
  const q = question.trim();
  if (!q.endsWith("?") || /\b(or|and)\b/i.test(q) || /അല്ലെങ്കിൽ/.test(q)) return false;
  return YES_NO_EN.test(q) || YES_NO_ML.test(q);
}
