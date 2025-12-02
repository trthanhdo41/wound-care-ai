from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, User, Message, Conversation
from datetime import datetime
import os
from werkzeug.utils import secure_filename

chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')

@chat_bp.route('/conversations', methods=['GET'])
@jwt_required()
def get_conversations():
    """Get all conversations for current user"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        # Get conversations where user is participant
        conversations = Conversation.query.filter(
            (Conversation.patient_id == user_id) | (Conversation.doctor_id == user_id)
        ).order_by(Conversation.updated_at.desc()).all()
        
        result = []
        for conv in conversations:
            # Get other participant
            other_user_id = conv.doctor_id if user.role == 'patient' else conv.patient_id
            other_user = User.query.get(other_user_id)
            
            # Get last message
            last_message = Message.query.filter_by(conversation_id=conv.id).order_by(Message.created_at.desc()).first()
            
            result.append({
                'id': conv.id,
                'other_user': {
                    'id': other_user.id,
                    'name': other_user.full_name,
                    'role': other_user.role
                },
                'last_message': {
                    'text': last_message.message_text if last_message else None,
                    'created_at': last_message.created_at.isoformat() if last_message else None
                },
                'unread_count': Message.query.filter_by(
                    conversation_id=conv.id,
                    is_read=False
                ).filter(Message.sender_id != user_id).count()
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/messages/<int:conversation_id>', methods=['GET'])
@jwt_required()
def get_messages(conversation_id):
    """Get all messages in a conversation"""
    try:
        user_id = int(get_jwt_identity())
        
        # Check if user is participant
        conversation = Conversation.query.get(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        if conversation.patient_id != user_id and conversation.doctor_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Get messages
        messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.created_at.asc()).all()
        
        # Mark messages as read
        Message.query.filter_by(conversation_id=conversation_id, is_read=False).filter(
            Message.sender_id != user_id
        ).update({'is_read': True})
        db.session.commit()
        
        result = [{
            'id': msg.id,
            'sender_id': msg.sender_id,
            'message_text': msg.message_text,
            'image_path': msg.image_path,
            'created_at': msg.created_at.isoformat(),
            'is_read': msg.is_read
        } for msg in messages]
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/send', methods=['POST'])
@jwt_required()
def send_message():
    """Send a message (text or image)"""
    try:
        user_id = int(get_jwt_identity())
        
        # Check if it's multipart/form-data (image upload) or JSON (text only)
        if request.content_type and 'multipart/form-data' in request.content_type:
            receiver_id = request.form.get('receiver_id')
            message_text = request.form.get('message_text', '')
            image_file = request.files.get('image')
            
            if not receiver_id:
                return jsonify({'error': 'Missing receiver_id'}), 400
            
            receiver_id = int(receiver_id)
            image_path = None
            
            # Upload image to ImgBB if provided
            if image_file:
                import base64
                import requests
                
                # Read image and encode to base64
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Upload to ImgBB
                imgbb_api_key = 'ae21ac039240a7d40788bcda9a822d8e'
                imgbb_url = 'https://api.imgbb.com/1/upload'
                
                payload = {
                    'key': imgbb_api_key,
                    'image': image_data
                }
                
                response = requests.post(imgbb_url, data=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    image_path = result['data']['url']
                else:
                    return jsonify({'error': 'Failed to upload image'}), 500
        else:
            data = request.get_json()
            receiver_id = data.get('receiver_id')
            message_text = data.get('message_text', '')
            image_path = None
            
            if not receiver_id:
                return jsonify({'error': 'Missing receiver_id'}), 400
        
        if not message_text and not image_path:
            return jsonify({'error': 'Message must contain text or image'}), 400
        
        user = User.query.get(user_id)
        receiver = User.query.get(receiver_id)
        
        if not receiver:
            return jsonify({'error': 'Receiver not found'}), 404
        
        # Check if conversation exists
        if user.role == 'patient':
            conversation = Conversation.query.filter_by(
                patient_id=user_id,
                doctor_id=receiver_id
            ).first()
        else:
            conversation = Conversation.query.filter_by(
                patient_id=receiver_id,
                doctor_id=user_id
            ).first()
        
        # Create conversation if not exists
        if not conversation:
            if user.role == 'patient':
                conversation = Conversation(patient_id=user_id, doctor_id=receiver_id)
            else:
                conversation = Conversation(patient_id=receiver_id, doctor_id=user_id)
            db.session.add(conversation)
            db.session.flush()
        
        # Create message
        message = Message(
            conversation_id=conversation.id,
            sender_id=user_id,
            message_text=message_text,
            image_path=image_path
        )
        db.session.add(message)
        
        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'message': 'Message sent',
            'data': {
                'id': message.id,
                'conversation_id': conversation.id,
                'sender_id': message.sender_id,
                'message_text': message.message_text,
                'image_path': image_path,
                'created_at': message.created_at.isoformat()
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@chat_bp.route('/conversations/<int:conversation_id>', methods=['DELETE'])
@jwt_required()
def delete_conversation(conversation_id):
    """Delete a conversation and all its messages"""
    try:
        user_id = int(get_jwt_identity())
        
        # Get conversation
        conversation = Conversation.query.get(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Check if user is participant
        if conversation.patient_id != user_id and conversation.doctor_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Delete all messages in conversation
        Message.query.filter_by(conversation_id=conversation_id).delete()
        
        # Delete conversation
        db.session.delete(conversation)
        db.session.commit()
        
        return jsonify({'message': 'Conversation deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
