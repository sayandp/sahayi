# Sahayi demo video: shot list (about 2 min 45 s)

**Setup:** run `python tasks.py up`, use a browser window at about 1280 × 800, and zoom the page to
125% so the Malayalam is readable. Type each question fresh, because the first answer is the one
that streams. Do a full rehearsal first and record only after it goes cleanly.

**Avoid the free-tier limit:** Groq allows only one or two qwen answers a minute. Leave about a
minute between questions, or edit the pauses out. If the "busy" notice appears, wait a minute and
ask again.

**Do not record while Gate 2 is running:** it uses the same Groq quota, so answers may be slower
or show the busy notice. Check with `Get-Content D:\rag\data\logs\auto_resume.log -Tail 5`.

| # | time | shot | exactly what to do | say over it |
|---|---|---|---|---|
| 1 | 0:00–0:15 | Home screen | Show the page. Click the ML/EN toggle once. | "Kerala residents can apply for many government schemes, but the rules sit in scattered PDFs and web pages, some of them in Malayalam. Sahayi tells you what you can get, and shows you the page that says so." |
| 2 | 0:15–0:40 | Malayalam question | Type: `പിഎം യശസ്വി പ്രീമെട്രിക് സ്കോളർഷിപ്പ് ഏതൊക്കെ ക്ലാസുകളിലെ കുട്ടികൾക്കാണ്?` | "A question in Malayalam. It answers in Malayalam: classes nine and ten. Sources appear first, then the answer streams in, and every sentence carries a citation." |
| 3 | 0:40–1:00 | Manglish question | Type: `pension apeksha ethra divasathinullil theerppu kalpikkanam?` | "Most people type Malayalam in English letters. Sahayi understands that too, and answers in Malayalam script: within 45 days." |
| 4 | 1:00–1:40 | English eligibility, then a follow-up | Type: `My husband passed away last year. Can I get the Indira Gandhi National Widow Pension?` When the "One more thing" box asks for age and BPL status, reply: `I am 35 years old and we are BPL.` | "Here's the real job: can I get this? It doesn't guess. It asks for what it's missing, my age and BPL status. Then it checks the rule: this pension starts at age 40, so at 35 the answer is no, and it says why." |
| 5 | 1:40–2:05 | Citation to the PDF page | Type: `Ente OBC list employmentinu use cheyyumo?` Click the citation on the first sentence, then the link to the source. The PDF opens at page 9. | "Don't trust me, check it. Each citation opens the exact page of the government document. Here's page 9, the State OBC List." |
| 6 | 2:05–2:25 | Abstain | Type: `How do I link my PAN card with Aadhaar?` | "This one isn't in the scheme documents. A normal chatbot would answer from memory. Sahayi says it doesn't know, and points you to a department to ask." |
| 7 | 2:25–2:45 | Close | Scroll to the "as of" date on any answer, then cut to the Gate 2 table in the README. | "Every answer shows how old its source is. The system is measured against 171 test questions, and those numbers are in the README. It's guidance, not legal advice: the department has the final word." |

## Why these questions

Each one worked in the eval, but check the notes before relying on it:

| shot | gold id | evidence |
|---|---|---|
| Malayalam | q034 | qwen, prompt v6: 6 of 6 statements supported. **gpt-oss-20b answered this one badly** (it gave the allowance, not the classes), so use this question only if qwen is the primary, or replace it after Gate 2. |
| Manglish | q063 | Prompt v6: both generators gave 45 days, all statements supported. |
| Eligibility | e007, then e006 | Prompt v4 (older), gpt-oss-20b and qwen. e007 did ask the follow-up. e006 got "not eligible, age 40–79" when asked in one sentence. **The two-step version shown here has not been run**, so check it in the rehearsal. |
| Citation | h001 | Prompt v6: qwen cites page 9 of a Kerala government PDF, all statements supported. gpt-oss-20b also cites page 9, but with a weaker judged result, so check it if gpt-oss-20b becomes primary. |
| Abstain | n014 | Prompt v4: abstained correctly. The contact it suggests comes from the nearest scheme and can look unrelated (Housing & Urban Affairs here), so don't read it aloud. |

**Re-check after Gate 2 finishes:** the final run files will show whether these items still pass on
the chosen generator.
