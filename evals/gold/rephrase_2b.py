"""One-off (Phase 2b): rephrase leaky gold questions into citizen-style wording.

Flagged by evals/leakage.py (BM25 rank 1 and > 40% word overlap with the gold text).
Only the question text changes; lang, schemes and evidence (hence gold) stay the same.
The original wording is kept in the source YAML as `question_orig` for traceability.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SRC = Path(__file__).with_name("questions_src.yaml")

REPHRASED = {
    "q001": "I'm a small farmer with my own land. How much will the central government put in my bank account every year, and in how many parts?",
    "q005": "If I join the farmers' pension scheme now, how much will I get every month once I'm old?",
    "q007": "How much do I have to pay from my side to insure my paddy crop this monsoon season?",
    "q009": "I lease land and farm it. Can I get a farm loan card from the bank?",
    "q011": "I sell fish in the market. Is there any central help for people like me?",
    "q013": "How much money will the government give me to build a house in my village?",
    "q014": "How do they decide who gets a rural government house first?",
    "q016": "Our family earns about 2.5 lakh a year. Which category would we be in for the city housing subsidy?",
    "q018": "My wife will deliver at the government hospital. How much cash help do we get after delivery?",
    "q020": "I'm pregnant with my first baby and work as a daily wage labourer. Is there any payment for me?",
    "q022": "My father is 72 and we are BPL. How much old age pension does the central scheme give?",
    "q025": "My husband, who earned for the family, died last month. Is there one-time help for us?",
    "q031": "We are Christians. Can my son in class 6 get a scholarship?",
    "q032": "minority kuttikalkku scholarship kittunnathu ethu class muthal ethu class vare aanu?",
    "q033": "My daughter is in class 9 and our family income is about 2 lakh. Can she get the central OBC scholarship?",
    "q037": "I want to start a small factory. Do they check my family income before giving the subsidy loan?",
    "q039": "I need about 40,000 rupees to start a tailoring unit. Which small business loan fits that?",
    "q041": "I sell vegetables on the roadside. How much loan can I get without security?",
    "q043": "I'm a carpenter. If I register under the artisan scheme, do they give any training?",
    "q045": "I applied for job card work. How soon should the panchayat give me work?",
    "q047": "We still cook with firewood and have no gas connection. Can I get a free one?",
    "q048": "puthiya gas connection edukkumbol aadyathe cylinder nirakkunnathinu panam kodukkano?",
    "q050": "എന്റെ ഭർത്താവ് മരിച്ചു. വീട്ടിലെ വരുമാനം കുറവാണ്, എനിക്ക് പെൻഷൻ കിട്ടാൻ വരുമാനം എത്ര വരെ ആകാം?",
    "q051": "ഭർത്താവ് മരിച്ച സ്ത്രീക്ക് പെൻഷന് അപേക്ഷിക്കാൻ എന്തൊക്കെ കടലാസുകൾ കൊണ്ടുപോകണം?",
    "q057": "ഞാൻ വർഷങ്ങളായി പാടത്ത് പണിയെടുക്കുന്നു. പെൻഷൻ കിട്ടാൻ എത്ര കാലം പണി ചെയ്തിരിക്കണം?",
    "q061": "പഞ്ചായത്തിൽ നിന്ന് കിട്ടുന്ന പെൻഷൻ മാസം എത്ര രൂപയാണ്?",
    "q062": "അമ്മയ്ക്ക് പെൻഷൻ വേണം. അപേക്ഷ എവിടെയാണ് കൊടുക്കേണ്ടത്?",
    "q065": "സർക്കാർ ആരോഗ്യ ഇൻഷുറൻസ് കാർഡ് ഉണ്ടെങ്കിൽ ആശുപത്രിയിൽ കിടന്നാൽ വർഷം എത്ര രൂപ വരെ സൗജന്യമാണ്?",
    "q070": "കേൾവിക്കുറവുള്ള കുഞ്ഞിന് സർക്കാർ ഓപ്പറേഷൻ ചെയ്തു കൊടുക്കുമോ?",
    "q071": "Our baby can't hear. Up to what age can we get the free implant surgery?",
    "q072": "My 10-year-old has leukaemia and we are poor. Is treatment free anywhere?",
    "q074": "My child has a heart problem and we can't afford treatment. Will the state help?",
    "q080": "My brother in Kasaragod is bedridden because of the pesticide spraying. Does he get monthly help?",
    "q081": "My son is deaf and goes to school. Is there a state scholarship for him?",
    "q083": "I'm blind. Can the government give me a talking phone or a reading device?",
    "q085": "I'm disabled and want to marry off my daughter. How old must she be to get the marriage help?",
    "q086": "നെൽകർഷകർക്ക് പെൻഷൻ കൊടുക്കാൻ തുടങ്ങിയത് എന്ന് മുതലാണ്?",
    "q089": "വെള്ളപ്പൊക്കത്തിൽ കൃഷി നശിച്ചാൽ സർക്കാർ ഇൻഷുറൻസ് കിട്ടാൻ എന്താണ് ചെയ്യേണ്ടത്?",
    "q090": "If rain fails or there is too much heat, does the state insurance pay farmers?",
    "q091": "Is there emergency money for farmers when pests or floods destroy crops?",
    "q094": "ഞാൻ തയ്യൽ ജോലി ചെയ്യുന്നു. വയസ്സായാൽ ബോർഡിൽ നിന്ന് മാസം എത്ര രൂപ കിട്ടും?",
    "q096": "My grandfather is a goldsmith over 60. Is there a monthly pension for traditional craftsmen?",
    "q097": "I'm in a women's self-help group. Can it help me start a small business?",
    "q098": "സ്വന്തമായി വീടില്ല. ലൈഫിൽ വീട് കിട്ടാൻ ആരൊക്കെയാണ് അർഹർ?",
    "q100": "ആദിവാസി കുടുംബത്തിന് ഇ.എം.എസ് പദ്ധതിയിൽ വീടിന് എത്ര രൂപ കിട്ടിയിരുന്നു?",
    "q102": "ഞാൻ എസ്.സി വിഭാഗത്തിൽ പെട്ട ഡിഗ്രി വിദ്യാർത്ഥിയാണ്. പഠനസഹായത്തിന്റെ നിയമങ്ങൾ എവിടെ നോക്കാം?",
    "q106": "പട്ടികജാതി കുട്ടികൾക്ക് പഠിക്കാൻ കമ്പ്യൂട്ടർ സൗജന്യമായി കിട്ടുമോ?",
    "q108": "ഞാൻ ദളിത് യുവാവാണ്. സ്വന്തമായി ഒരു കട തുടങ്ങാൻ വകുപ്പിൽ നിന്ന് സഹായം കിട്ടുമോ?",
    "q111": "ഞാൻ ആദിവാസിയാണ്, മറ്റൊരു ജാതിയിലുള്ളയാളെ വിവാഹം കഴിച്ചു. ഞങ്ങൾക്ക് എന്തെങ്കിലും സഹായം ഉണ്ടോ?",
    "q113": "പിന്നാക്ക വിഭാഗത്തിലെ കുട്ടികൾക്കുള്ള പുതിയ സ്കോളർഷിപ്പിന് അപേക്ഷ എങ്ങനെ കൊടുക്കും?",
    "q114": "My son is in class 6 and we belong to an OEC community. How much study help will he get?",
    "q117": "അച്ഛൻ മരിച്ചുപോയി. ഞാൻ നഴ്സിംഗിന് പഠിക്കുന്ന ഈഴവ പെൺകുട്ടിയാണ്. എനിക്ക് പ്രത്യേക സഹായം കിട്ടുമോ?",
    "q118": "Is my caste counted as backward in Kerala? Where is the list?",
    "q122": "പിന്നാക്ക വിഭാഗത്തിൽ പെട്ട എനിക്ക് വിദേശത്ത് പഠിക്കാൻ സർക്കാർ സഹായം ഈ വർഷം കിട്ടുമോ?",
    "q124": "I just enrolled as a lawyer and I'm from a backward community. Is there any monthly help to start practice?",
    "q125": "I'm a Muslim girl starting my degree. Is there a state scholarship I can apply for?",
}

# Second pass on the 11 still flagged after the first rewrite.
REPHRASED.update(
    {
        "q007": "Rain may ruin my rice this year. What will crop insurance cost me?",
        "q011": "My family catches and sells fish. Does Delhi have some help for fishermen?",
        "q018": "We are poor and my wife's baby is due at the PHC next month. Will we get any money?",
        "q022": "My father is 72 and we have a yellow ration card. What does the old people's pension pay?",
        "q025": "My husband died last month and he was the only one earning. Can we get lump-sum help?",
        "q043": "I make wooden furniture by hand. Will the government teach me new skills and pay me while I learn?",
        "q065": "ആശുപത്രി ചെലവ് താങ്ങാൻ പറ്റുന്നില്ല. സർക്കാരിന്റെ കാർഡ് ഉള്ളവർക്ക് എത്ര രൂപ വരെ ചികിത്സ ഫ്രീ ആണ്?",
        "q071": "Our baby can't hear anything. Is there an age limit for the free ear operation?",
        "q083": "I can't see. Will the state give me a phone that reads aloud?",
        "q114": "My son is in class 6 and we are Latin Catholics. How much yearly study help can he get?",
        "q124": "I just became a lawyer and I'm Ezhava. Will the state pay me something while I start out?",
    }
)


def main() -> None:
    data = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    changed = 0
    for q in data["questions"]:
        new = REPHRASED.get(q["id"])
        if new and q["question"] != new:
            q.setdefault("question_orig", q["question"])
            q["question"] = new
            changed += 1
    header = SRC.read_text(encoding="utf-8").split("questions:")[0]
    body = yaml.safe_dump({"questions": data["questions"]}, allow_unicode=True, sort_keys=False,
                          width=4096)  # fmt: skip
    SRC.write_text(header + body, encoding="utf-8")
    print(f"rephrased {changed} questions")


if __name__ == "__main__":
    main()
