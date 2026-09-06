# 🛒 Enterprise Hybrid RAG: Vector RAG + Graph RAG with GPT

> **Vector RAG finds the right paragraph. Graph RAG connects the relationships. Hybrid RAG does both.**

A complete, runnable reference implementation of **Hybrid Retrieval-Augmented Generation** for the enterprise,
explained with a supermarket chain (*DailyBasket*). It combines semantic search over policy documents with
multi-hop traversal over a knowledge graph, then uses a **GPT model** to write a grounded answer with citations.

> 🚧 **Work in progress.** Building this over weekends.

---

## 💡 Why hybrid?

| Question | What it needs | Lane |
|---|---|---|
| *"How many sick leaves do store staff get?"* | One paragraph in the HR policy | 📄 **Vector RAG** |
| *"FreshDairy milk is recalled. Which stores and managers are affected?"* | Supplier → Product → Store → Manager links. No single document has this | 🕸️ **Graph RAG** |
| *"Milk recall: what must each affected store manager do?"* | **Who** is affected (graph) + **what to do** (Recall SOP) | 🤝 **Hybrid RAG** |

Vector-only RAG answers the first question well and struggles with the other two. Most real enterprise
questions look like the third one.

---

## 📄 License

MIT © 2026 **Sudharsan SambathKumar**
