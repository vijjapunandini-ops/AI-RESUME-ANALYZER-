from processors.document_processor import extract_document_text


def main():
    filename = "sample.pdf"

    with open(filename, "rb") as file:
        file_bytes = file.read()

    result = extract_document_text(
        filename,
        file_bytes,
    )

    print("=" * 60)
    print("DOCUMENT:", result["filename"])
    print("TYPE:", result["extension"])
    print("WORDS:", result["word_count"])
    print("CHARACTERS:", result["character_count"])
    print("=" * 60)

    print(result["text"][:2000])


if __name__ == "__main__":
    main()