from flask import Blueprint, send_file, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, AnalysisResult, User
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import os
from datetime import datetime

pdf_bp = Blueprint('pdf', __name__, url_prefix='/api/analysis')

@pdf_bp.route('/export-pdf/<int:analysis_id>', methods=['GET'])
@jwt_required()
def export_pdf(analysis_id):
    """Export analysis result as PDF"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        print(f"📄 PDF Export - User ID: {user_id}, Role: {user.role}")
        
        # Get analysis result
        analysis = AnalysisResult.query.get(analysis_id)
        if not analysis:
            print(f"❌ Analysis {analysis_id} not found")
            return jsonify({'error': 'Analysis not found'}), 404
        
        print(f"📄 Analysis {analysis_id} belongs to patient_id: {analysis.patient_id}")
        
        # Check permission - allow if:
        # 1. User is a doctor or admin (can see all)
        # 2. User is a patient and the analysis belongs to their patient record
        if user.role in ['doctor', 'admin']:
            # Doctors and admins can access all analyses
            pass
        elif user.role == 'patient':
            # Patient can only access their own analyses
            if not user.patient or analysis.patient_id != user.patient.id:
                print(f"❌ Permission denied - User {user_id} (patient.id={user.patient.id if user.patient else 'None'}) cannot access analysis of patient {analysis.patient_id}")
                return jsonify({'error': 'Unauthorized - This analysis belongs to another patient'}), 403
        else:
            print(f"❌ Permission denied - Unknown role: {user.role}")
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0ea5e9'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Title
        elements.append(Paragraph("Diabetic Foot Ulcer Analysis Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Patient Info
        patient = User.query.get(analysis.patient_id)
        info_data = [
            ['Patient ID:', str(patient.id) if patient else 'N/A'],
            ['Patient Name:', patient.full_name if patient and patient.full_name else 'N/A'],
            ['Analysis Date:', analysis.created_at.strftime('%Y-%m-%d %H:%M') if analysis.created_at else 'N/A'],
            ['Analysis ID:', str(analysis.id)]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Risk Assessment
        elements.append(Paragraph("Risk Assessment", heading_style))
        
        # Handle both old and new data structures
        if hasattr(analysis, 'results') and analysis.results:
            results = analysis.results
            risk = results.get('risk_assessment', {})
        else:
            # Fallback to direct fields
            risk = {
                'risk_level': analysis.risk_level or 'N/A',
                'risk_score': float(analysis.risk_score) if analysis.risk_score else 0,
                'recommendation': 'Please consult with your healthcare provider',
                'care_guidelines': []
            }
            results = {'color_analysis': analysis.color_analysis or {}, 'size_metrics': {}}
        
        risk_level = risk.get('risk_level', 'N/A').upper()
        risk_score = risk.get('risk_score', 0)
        
        risk_color = colors.green if risk_level == 'LOW' else (colors.orange if risk_level == 'MEDIUM' else colors.red)
        
        risk_data = [
            ['Risk Level:', risk_level],
            ['Risk Score:', f"{risk_score}/100"],
            ['Recommendation:', risk.get('recommendation', 'N/A')]
        ]
        
        risk_table = Table(risk_data, colWidths=[2*inch, 4*inch])
        risk_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (1, 0), (1, 0), risk_color),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (1, 0), (1, 0), 14),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(risk_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Color Analysis
        elements.append(Paragraph("Color Analysis", heading_style))
        
        color_analysis = results.get('color_analysis', {})
        pixel_pct = color_analysis.get('pixel_based_percentages', {})
        
        color_data = [
            ['Red:', f"{pixel_pct.get('Red', 0):.1f}%"],
            ['Yellow:', f"{pixel_pct.get('Yellow', 0):.1f}%"],
            ['Dark:', f"{pixel_pct.get('Dark', 0):.1f}%"]
        ]
        
        color_table = Table(color_data, colWidths=[2*inch, 4*inch])
        color_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(color_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Size Metrics
        size_metrics = results.get('size_metrics', {})
        if size_metrics:
            elements.append(Paragraph("Size Metrics", heading_style))
            size_data = [
                ['Area:', f"{size_metrics.get('area_cm2', 0):.2f} cm²"],
                ['Perimeter:', f"{size_metrics.get('perimeter_cm', 0):.2f} cm"],
                ['Width:', f"{size_metrics.get('width_cm', 0):.2f} cm"],
                ['Height:', f"{size_metrics.get('height_cm', 0):.2f} cm"]
            ]
            
            size_table = Table(size_data, colWidths=[2*inch, 4*inch])
            size_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(size_table)
        
        # Care Guidelines
        care_guidelines = risk.get('care_guidelines', [])
        if care_guidelines:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("AI-Generated Care Guidelines", heading_style))
            
            for guideline in care_guidelines:
                bullet = Paragraph(f"• {guideline.get('text', '')}", styles['Normal'])
                elements.append(bullet)
                elements.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        # Return PDF file
        filename = f"analysis_report_{analysis_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Error exporting PDF: {e}")
        return jsonify({'error': str(e)}), 500
