# AI pairing playbook (how to use AI on camera)

Tools: **Claude 4.6 / ChatGPT 4.5 / Gemini**, single-assistant, Cursor-like, **no
multi-agent**. They're evaluating *how you use AI*, not whether you can.

## The golden rule
**Narrate + verify.** Every time you use AI: say what you're asking, then read the
output aloud and judge it. The moment you catch a mistake is the moment you pass.

## Prompt patterns that look senior
- **Orient:** "Summarize this repo's architecture and where requests enter. List the
  main modules and their dependencies." → then open the files to confirm.
- **Explain:** "Walk me through what this function does and its edge cases."
- **Scaffold:** "Draft a function with this signature that does X. Don't change other
  files." → review the diff.
- **Test:** "Write a unit test for this covering the empty-input and failure cases."
- **Review:** "What bugs, security issues, or missing error handling are in this diff?"
- **Constrain:** give it the interfaces/types and the existing style to match.

## What to say out loud (scripts)
- "I'll have the assistant draft this, but I want to verify the API actually exists."
- "It suggested X — that's wrong because Y; I'll adjust."
- "Good scaffold, but it's missing a timeout and input validation. Adding those."
- "I won't accept this blindly — let me trace how it handles the error path."

## Anti-patterns (instant red flags)
- Pasting a big AI diff you can't explain line-by-line.
- Letting AI pick the architecture without your judgment.
- Going silent while AI works.
- Trusting an API/library the model invented (verify imports exist).
- Using AI to skip understanding the problem (clarify with the *human* first).

## When AI and you disagree
Trust your reasoning + the actual code/tests over the model. Say why. That's exactly
the judgment they want to see.

## Time management with AI
AI makes you faster at typing, not at *thinking*. Spend the saved time on: clarifying,
edge cases, tests, and explaining trade-offs. "We evaluate systems thinking, not speed."
