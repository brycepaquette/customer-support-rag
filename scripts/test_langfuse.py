from anthropic import Anthropic

from customer_support_rag.classifier import classify_ticket

client = Anthropic()

tickets = [
    "PyCelonis API is returning 500 errors",
    "How do I export data from a data pool?",
    "I need access to the admin console",
]

for ticket in tickets:
    result = classify_ticket(client, ticket)
    print(f"Q: {ticket}")
    print(f"A: {result}\n")
