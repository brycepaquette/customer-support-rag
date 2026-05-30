from src.customer_support_rag.client import get_client


def main():
    client = get_client()
    print("Client initialized:", client)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What is PyCelonis?"}],
    )
    print(
        message.content[0].text,
        message.usage.input_tokens,
        message.usage.output_tokens,
        message.usage.total_tokens,
    )


if __name__ == "__main__":
    main()
