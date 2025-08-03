# AI Agent Memory Types

To build useful and intelligent agents, it's important to equip them with different types of memory. Each memory type supports a specific role in how the agent thinks, acts, and adapts.

---

## 1. Temporary Memory (Short-Term)

- **Meaning**: This holds recent information the agent is working with.
- **Purpose**: Helps the agent follow a conversation or task without losing track of what's happening now.
- **How to Use**: Store the latest few exchanges or key facts in a session. This can be in a buffer or just in memory while the program is running.

---

## 2. Experience-Based Memory (Episodic)

- **Meaning**: Stores records of specific situations the agent has dealt with before.
- **Purpose**: Helps the agent refer back to past interactions or tasks—for example, remembering what it did last time something similar happened.
- **How to Use**: Save entries with timestamps and details after each session. Could be saved in files or a database.

---

## 3. Factual Memory (Semantic)

- **Meaning**: Holds general facts, definitions, and other static information.
- **Purpose**: Allows the agent to access known information quickly—like how something works or what a term means.
- **How to Use**: Build a simple lookup system, or store in a vector index or key-value store.

---

## 4. Task Procedure Memory (Procedural)

- **Meaning**: Contains instructions or routines for tasks the agent knows how to perform.
- **Purpose**: Enables the agent to repeat tasks correctly, such as setting up a server or creating a report.
- **How to Use**: Save these as scripts, config files, or structured workflows.

---

## 5. Persistent Memory (Long-Term)

- **Meaning**: Combines everything the agent should remember across sessions—its experiences, facts, and how-to guides.
- **Purpose**: Ensures the agent builds on what it learns over time and doesn’t start fresh every time it runs.
- **How to Use**: Use a file system or database to store all memory types for long-term use.

---

## Overview Table

| Memory Type  | Span         | Content Stored         | Key Function                      |
|--------------|--------------|------------------------|------------------------------------|
| Short-Term   | Current use  | Recent info/convo      | Keeps the agent focused            |
| Episodic     | Past events  | Situational history     | Reflects on specific experiences   |
| Semantic     | Timeless     | Facts and definitions   | Provides fast, factual answers     |
| Procedural   | Reusable     | Step-by-step actions    | Guides how tasks are completed     |
| Long-Term    | Ongoing      | All above combined      | Builds learning and continuity     |

---

## For example: Putting All Together

- When the agent starts a task, it reads from temporary, factual, and experience memories to get oriented.
- During its operation, it updates short-term and records key events.
- After finishing, it logs what it did, updates facts if needed, and saves any changes to its long-term memory.

---

## Why This Matters

Organizing memory in this way helps the agent operate more like a reliable teammate. It keeps track of its work, doesn’t repeat mistakes, and can be trusted to remember what it’s supposed to—both now and in the future.

