"""
services/invoice_service.py - Corporate B2B PDF Invoice Generator for ApexSales AI
Generates professional vector PDF invoices for closed sales packages.
"""

import os
import io
import time
from typing import Dict, Any, Optional
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT


PACKAGES_METADATA = {
    "starter": {
        "name": "ApexSales AI — Starter Plan",
        "description": "Autonomous AI B2B Voice SDR (Up to 5,000 call minutes/month, CRM integration, Sub-300ms SLA)",
        "amount": 499.00
    },
    "growth": {
        "name": "ApexSales AI — Growth Plan",
        "description": "Multi-Voice Autonomous Outbound Engine (Up to 25,000 mins/mo, K-Means Clustering, ChromaDB RAG, Sub-300ms SLA)",
        "amount": 1499.00
    },
    "enterprise": {
        "name": "ApexSales AI — Enterprise Custom",
        "description": "Dedicated Enterprise AI Pipeline (Unlimited minutes, Sub-200ms dedicated VPC, Custom Voice Cloning & SOC2 SLA)",
        "amount": 3500.00
    }
}


def generate_invoice_pdf(
    plan_key: str = "growth",
    amount: Optional[float] = None,
    customer_name: str = "Valued Customer",
    company_name: str = "Prospect Organization",
    email: Optional[str] = "contact@client.com",
    phone: Optional[str] = None,
    invoice_number: Optional[str] = None
) -> bytes:
    """
    Generates an executive-ready B2B Sales Invoice in PDF format.
    Returns bytes of the generated PDF.
    """
    buffer = io.BytesIO()

    # Document Setup
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    # Normalize package info
    plan_key_clean = str(plan_key).lower()
    if "499" in plan_key_clean or "starter" in plan_key_clean:
        pkg = PACKAGES_METADATA["starter"]
    elif "3500" in plan_key_clean or "enterprise" in plan_key_clean:
        pkg = PACKAGES_METADATA["enterprise"]
    else:
        pkg = PACKAGES_METADATA["growth"]

    plan_title = pkg["name"]
    plan_desc = pkg["description"]
    final_amount = float(amount) if amount is not None else float(pkg["amount"])

    if not invoice_number:
        timestamp_code = int(time.time()) % 100000
        invoice_number = f"INV-2026-{timestamp_code:05d}"

    issue_date = datetime.now().strftime("%B %d, %Y")
    due_date = "Due Upon Receipt (Net 15)"

    styles = getSampleStyleSheet()

    # Custom typography
    royal_color = colors.HexColor("#1e3a8a")
    dark_text = colors.HexColor("#0f172a")
    muted_text = colors.HexColor("#64748b")
    brand_blue = colors.HexColor("#2563eb")
    table_header_bg = colors.HexColor("#eff6ff")
    table_border = colors.HexColor("#cbd5e1")

    title_style = ParagraphStyle(
        "InvoiceTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=royal_color
    )

    subtitle_style = ParagraphStyle(
        "InvoiceSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=muted_text
    )

    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=royal_color
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=14,
        textColor=dark_text
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=royal_color
    )

    right_align_style = ParagraphStyle(
        "RightAlign",
        parent=body_style,
        alignment=TA_RIGHT
    )

    right_bold_style = ParagraphStyle(
        "RightBold",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=brand_blue,
        alignment=TA_RIGHT
    )

    elements = []

    # 1. Header Grid: Brand on Left, Invoice Details on Right
    header_data = [
        [
            Paragraph("<b>APEXSALES AI</b><br/><font color='#64748b' size='8'>Autonomous Real-Time B2B Voice Infrastructure<br/>San Francisco, CA & New York, NY<br/>support@apexsales.ai | +1 (888) 555-APEX</font>", subtitle_style),
            Paragraph(f"<font size='18' color='#1e3a8a'><b>INVOICE</b></font><br/><br/><b>Invoice #:</b> {invoice_number}<br/><b>Date:</b> {issue_date}<br/><b>Terms:</b> {due_date}", right_align_style)
        ]
    ]

    header_table = Table(header_data, colWidths=[3.5 * inch, 3.8 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(header_table)

    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=15))

    # 2. Bill To & Payment Summary
    bill_data = [
        [
            Paragraph("<b>BILLED TO:</b>", h2_style),
            Paragraph("<b>PACKAGE DETAILS:</b>", h2_style)
        ],
        [
            Paragraph(f"<b>{customer_name}</b><br/>{company_name}<br/>Email: {email or 'prospect@organization.com'}" + (f"<br/>Phone: {phone}" if phone else "") + "<br/>Account Status: <b>Active (Call Confirmed)</b>", body_style),
            Paragraph(f"<b>Service:</b> {plan_title}<br/><b>Billing Frequency:</b> Monthly Recurring<br/><b>Service SLA:</b> Sub-300ms Voice Response SLA<br/><b>Deployment:</b> Instant Activation", body_style)
        ]
    ]

    bill_table = Table(bill_data, colWidths=[3.5 * inch, 3.8 * inch])
    bill_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(bill_table)
    elements.append(Spacer(1, 15))

    # 3. Line Items Table
    items_data = [
        [
            Paragraph("ITEM & SERVICE DESCRIPTION", table_header_style),
            Paragraph("QTY", table_header_style),
            Paragraph("UNIT PRICE", table_header_style),
            Paragraph("TOTAL", table_header_style)
        ],
        [
            Paragraph(f"<b>{plan_title}</b><br/><font color='#64748b'>{plan_desc}</font>", body_style),
            Paragraph("1", body_style),
            Paragraph(f"${final_amount:,.2f}", body_style),
            Paragraph(f"<b>${final_amount:,.2f}</b>", body_style)
        ],
        [
            Paragraph("<b>🎁 15-Minute Complimentary Implementation Demo & POC Session</b><br/><font color='#64748b'>1-on-1 walkthrough with Solutions Architect, custom CRM prompt tuning & live test</font>", body_style),
            Paragraph("1", body_style),
            Paragraph("<font color='#059669'><b>FREE BONUS</b></font>", body_style),
            Paragraph("<font color='#059669'><b>$0.00</b></font>", body_style)
        ],
        [
            Paragraph("<b>Enterprise CRM Sync & Webhook Activation (HubSpot / Salesforce)</b><br/><font color='#64748b'>Complimentary real-time lead webhook integration & telemetry setup</font>", body_style),
            Paragraph("1", body_style),
            Paragraph("<font color='#2563eb'><b>INCLUDED</b></font>", body_style),
            Paragraph("<b>$0.00</b>", body_style)
        ]
    ]

    item_table = Table(items_data, colWidths=[4.2 * inch, 0.6 * inch, 1.2 * inch, 1.3 * inch])
    item_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), table_header_bg),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, table_border),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(item_table)
    elements.append(Spacer(1, 15))

    # 4. Totals Summary Table
    total_data = [
        ["", Paragraph("<b>Subtotal:</b>", right_align_style), Paragraph(f"${final_amount:,.2f}", right_align_style)],
        ["", Paragraph("<b>Estimated Tax (0%):</b>", right_align_style), Paragraph("$0.00", right_align_style)],
        ["", Paragraph("<b>Total Due:</b>", right_align_style), Paragraph(f"<b>${final_amount:,.2f} USD</b>", right_bold_style)]
    ]

    total_table = Table(total_data, colWidths=[4.2 * inch, 1.6 * inch, 1.5 * inch])
    total_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(total_table)
    elements.append(Spacer(1, 25))

    # 5. Payment Terms & Assurance Notes
    notes_html = f"""
    <b>PAYMENT & SERVICE TERMS:</b><br/>
    1. <b>Payment Methods:</b> Wire Transfer, ACH, or Corporate Credit Card via ApexSales Secure Portal.<br/>
    2. <b>SLA & Performance Guarantee:</b> 99.9% Uptime with sub-300ms speech-to-speech dialogue latency.<br/>
    3. <b>Complimentary 15-Min Live Demo:</b> Includes a 1-on-1 technical setup and pilot demo session with an ApexSales Solutions Architect at $0.00 cost.<br/>
    4. <b>Cancellation:</b> Flexible 30-day notice with zero hidden cancellation fees.<br/>
    5. <b>Dedicated Executive:</b> Account managed by Marcus Vance (Enterprise Lead) & Sarah (AI Lead AE).
    """
    elements.append(Paragraph(notes_html, subtitle_style))
    elements.append(Spacer(1, 25))

    # 6. Authorized Signature Footer
    sig_data = [
        [
            Paragraph("<b>Authorized Representative:</b><br/><font color='#2563eb' size='11'><i>Marcus Vance</i></font><br/>Senior Enterprise Director, ApexSales AI", body_style),
            Paragraph("<b>Client Confirmation:</b><br/><font color='#64748b'>Confirmed during live sales call</font><br/>Digital Acceptance Verified", right_align_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[3.6 * inch, 3.7 * inch])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(sig_table)

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
