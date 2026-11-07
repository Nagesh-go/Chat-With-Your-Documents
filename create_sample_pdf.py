"""
Sample PDF Generator
Creates a valid multi-page PDF using raw PDF byte structure.
No extra packages needed beyond what is already installed.
"""
import os
import struct

def create_sample_pdf(output_path: str):
    """
    Writes a syntactically valid multi-page PDF file to output_path.
    The PDF contains 3 pages of realistic text suitable for RAG testing.
    """
    pages_content = [
        (
            "Company Leave Policy - Page 1\n\n"
            "Annual Leave:\n"
            "Full-time employees are entitled to 20 days of paid annual leave per calendar year. "
            "Leave requests must be submitted at least 14 days in advance and approved by the "
            "department manager. Unused leave of up to 5 days may be carried over to the following year. "
            "Employees who resign forfeit any unused leave beyond statutory minimums.\n\n"
            "Sick Leave:\n"
            "Employees are entitled to 10 days of paid sick leave annually. "
            "A medical certificate is required for absences exceeding 3 consecutive days."
        ),
        (
            "Company Leave Policy - Page 2\n\n"
            "Maternity and Paternity Leave:\n"
            "Female employees are entitled to 26 weeks of paid maternity leave. "
            "Male employees are entitled to 2 weeks of paid paternity leave following the birth or "
            "adoption of a child. Leave must commence within 6 months of the child's birth date.\n\n"
            "Refund and Reimbursement Policy:\n"
            "Business travel expenses must be submitted within 30 days of incurring the cost. "
            "Reimbursements are processed within 10 working days. Receipts are mandatory for all "
            "expenses above $25. Personal expenses are not eligible for reimbursement."
        ),
        (
            "Company Leave Policy - Page 3\n\n"
            "Public Holidays:\n"
            "Employees are entitled to all gazetted public holidays. When a public holiday falls on "
            "a weekend, the following Monday is observed as a substitute holiday.\n\n"
            "Remote Work Policy:\n"
            "Employees may work remotely up to 3 days per week subject to manager approval. "
            "A stable internet connection and a dedicated workspace are required. "
            "Core working hours of 10:00 AM to 4:00 PM must be maintained regardless of location. "
            "Remote work arrangements are reviewed quarterly and may be adjusted based on business needs."
        ),
    ]

    # --- Build raw PDF structure ---
    pdf_lines = []
    offsets = []

    def add(line: str):
        pdf_lines.append(line.encode("latin-1") if isinstance(line, str) else line)

    # PDF Header
    add("%PDF-1.4\n")
    add("%\xe2\xe3\xcf\xd3\n")  # Binary comment (marks file as binary)

    page_obj_ids = []
    content_obj_ids = []

    # Catalog obj = 1, Pages obj = 2, page objs start at 3
    catalog_id = 1
    pages_id = 2
    next_id = 3

    page_text_streams = []
    for text in pages_content:
        # Escape special PDF chars
        escaped = (
            text.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
        )
        # Build a simple text stream with proper font and line wrapping
        lines = []
        for paragraph in escaped.split("\n"):
            # Split long lines manually at ~90 chars
            while len(paragraph) > 90:
                lines.append(paragraph[:90])
                paragraph = paragraph[90:]
            lines.append(paragraph)

        stream_lines = ["BT", "/F1 12 Tf", "50 750 Td", "14 TL"]
        for ln in lines:
            stream_lines.append(f"({ln}) Tj T*")
        stream_lines.append("ET")
        page_text_streams.append("\n".join(stream_lines))

    # Write content stream objects
    for stream_text in page_text_streams:
        content_id = next_id
        content_obj_ids.append(content_id)
        next_id += 1
        offsets.append(len(b"".join(pdf_lines)))
        encoded = stream_text.encode("latin-1")
        add(f"{content_id} 0 obj\n")
        add(f"<< /Length {len(encoded)} >>\n")
        add("stream\n")
        pdf_lines.append(encoded)
        add("\nendstream\nendobj\n\n")

    # Write page objects
    for i, content_id in enumerate(content_obj_ids):
        page_id = next_id
        page_obj_ids.append(page_id)
        next_id += 1
        offsets.append(len(b"".join(pdf_lines)))
        add(f"{page_id} 0 obj\n")
        add(f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\n")
        add("endobj\n\n")

    # Write Pages object (id=2)
    kids = " ".join([f"{pid} 0 R" for pid in page_obj_ids])
    pages_offset = len(b"".join(pdf_lines))
    add(f"{pages_id} 0 obj\n")
    add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_ids)} >>\n")
    add("endobj\n\n")

    # Write Catalog object (id=1)
    catalog_offset = len(b"".join(pdf_lines))
    add(f"{catalog_id} 0 obj\n")
    add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>\n")
    add("endobj\n\n")

    # Cross-reference table
    xref_offset = len(b"".join(pdf_lines))
    total_objects = next_id  # all ids from 0 to next_id-1

    all_offsets = [0] * total_objects
    # Content streams: ids 3..3+len-1
    for i, off in enumerate(offsets[:len(content_obj_ids)]):
        all_offsets[3 + i] = off
    # Page objects
    for i, off in enumerate(offsets[len(content_obj_ids):]):
        all_offsets[3 + len(content_obj_ids) + i] = off
    all_offsets[pages_id] = pages_offset
    all_offsets[catalog_id] = catalog_offset

    add(f"xref\n0 {total_objects}\n")
    add(f"0000000000 65535 f \n")  # Object 0 (free)
    for i in range(1, total_objects):
        add(f"{all_offsets[i]:010d} 00000 n \n")

    # Trailer
    add(f"trailer\n<< /Size {total_objects} /Root {catalog_id} 0 R >>\n")
    add(f"startxref\n{xref_offset}\n%%EOF\n")

    # Write to disk
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in pdf_lines:
            f.write(chunk)

    print(f"[SUCCESS] Created valid PDF: '{output_path}'")
    print(f"          Pages: {len(pages_content)} | Topics: Leave Policy, Refunds, Remote Work")

if __name__ == "__main__":
    output = os.path.join("data", "sample.pdf")
    create_sample_pdf(output)
    print("\nNow run:  python src/document_loader.py")