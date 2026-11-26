# Seeded faults (Exercise 2)

Agent: Qwen 3.5 9B (`bikeshare-coder`, Ollama, 64k context) in headless Claude Code
sessions. Each fault was seeded by Qwen through `/seed-fault` (so the edit went
through the Edit tool and the PostToolUse hook), then repaired by a *fresh*
Qwen session told only that `make test` was red. "Turns" is the session's
`num_turns` from the Claude Code result message. Transcripts: `logs/qwen/E2*`.
Output below is pasted from those runs, not paraphrased.
