import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import { FiSend, FiUser, FiMessageCircle, FiImage, FiX } from 'react-icons/fi';
import './ChatWindow.css';

export function ChatWindow() {
  const { API_URL, user } = useAuth();
  const [conversations, setConversations] = useState([]);
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [showDoctorList, setShowDoctorList] = useState(false);
  const [doctors, setDoctors] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchConversations();
    if (user?.role === 'patient') {
      fetchDoctors();
    }
  }, []);

  useEffect(() => {
    if (selectedConversation) {
      fetchMessages();
      const interval = setInterval(fetchMessages, 3000); // Poll every 3s
      return () => clearInterval(interval);
    }
  }, [selectedConversation]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchConversations = async () => {
    try {
      const response = await axios.get(`${API_URL}/chat/conversations`);
      setConversations(response.data);
      if (response.data.length > 0 && !selectedConversation) {
        setSelectedConversation(response.data[0]);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
    }
  };

  const fetchDoctors = async () => {
    try {
      const response = await axios.get(`${API_URL}/patients/doctors`);
      setDoctors(response.data);
    } catch (error) {
      console.error('Error fetching doctors:', error);
    }
  };

  const startChatWithDoctor = async (doctor) => {
    try {
      // Send first message to create conversation
      await axios.post(`${API_URL}/chat/send`, {
        receiver_id: doctor.id,
        message_text: 'Hello Doctor, I would like to consult with you.'
      });
      setShowDoctorList(false);
      fetchConversations();
    } catch (error) {
      console.error('Error starting chat:', error);
    }
  };

  const fetchMessages = async () => {
    if (!selectedConversation) return;
    try {
      const response = await axios.get(`${API_URL}/chat/messages/${selectedConversation.id}`);
      setMessages(response.data);
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const deleteConversation = async (conversationId) => {
    if (!window.confirm('Are you sure you want to delete this entire conversation? This cannot be undone.')) {
      return;
    }
    
    try {
      await axios.delete(`${API_URL}/chat/conversations/${conversationId}`);
      setSelectedConversation(null);
      fetchConversations(); // Refresh conversation list
    } catch (error) {
      console.error('Error deleting conversation:', error);
      alert('Failed to delete conversation');
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if ((!newMessage.trim() && !selectedImage) || !selectedConversation) return;

    setLoading(true);
    try {
      if (selectedImage) {
        // Send with image
        const formData = new FormData();
        formData.append('receiver_id', selectedConversation.other_user.id);
        formData.append('message_text', newMessage || '');
        formData.append('image', selectedImage);

        await axios.post(`${API_URL}/chat/send`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
      } else {
        // Send text only
        await axios.post(`${API_URL}/chat/send`, {
          receiver_id: selectedConversation.other_user.id,
          message_text: newMessage
        });
      }
      
      setNewMessage('');
      clearImage();
      fetchMessages();
      fetchConversations();
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-window-container">
      {/* Conversations List */}
      <div className="conversations-sidebar">
        <div className="sidebar-header-chat">
          <h3>Conversations</h3>
          {user?.role === 'patient' && (
            <button 
              className="btn-new-chat"
              onClick={() => setShowDoctorList(true)}
              title="Start new chat with doctor"
            >
              +
            </button>
          )}
        </div>
        {conversations.length === 0 ? (
          <div className="no-conversations">
            <FiMessageCircle className="no-conv-icon" />
            <p>No conversations yet</p>
            <p className="no-conv-hint">
              {user?.role === 'patient' 
                ? 'Click + to start chatting with a doctor' 
                : 'Wait for patients to message you'}
            </p>
          </div>
        ) : (
          <div className="conversations-list">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`conversation-item ${selectedConversation?.id === conv.id ? 'active' : ''}`}
              >
                <div className="conv-avatar" onClick={() => setSelectedConversation(conv)}>
                  <FiUser />
                </div>
                <div className="conv-info" onClick={() => setSelectedConversation(conv)}>
                  <h4>{conv.other_user.name}</h4>
                  <p className="conv-role">{conv.other_user.role}</p>
                  {conv.last_message?.text && (
                    <p className="conv-last-message">{conv.last_message.text.substring(0, 30)}...</p>
                  )}
                </div>
                <button 
                  className="btn-delete-conversation" 
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteConversation(conv.id);
                  }}
                  title="Delete conversation"
                >
                  <FiX />
                </button>
                {conv.unread_count > 0 && (
                  <span className="unread-badge">{conv.unread_count}</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Doctor List Modal */}
      {showDoctorList && (
        <div className="modal-overlay-chat" onClick={() => setShowDoctorList(false)}>
          <div className="modal-content-chat" onClick={(e) => e.stopPropagation()}>
            <h3>Select a Doctor</h3>
            <div className="doctors-list">
              {doctors.map((doctor) => (
                <div 
                  key={doctor.id} 
                  className="doctor-item"
                  onClick={() => startChatWithDoctor(doctor)}
                >
                  <div className="doctor-avatar">
                    <FiUser />
                  </div>
                  <div className="doctor-info">
                    <h4>{doctor.full_name}</h4>
                    <p>{doctor.email}</p>
                  </div>
                </div>
              ))}
            </div>
            <button 
              className="btn-close-modal"
              onClick={() => setShowDoctorList(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Chat Messages */}
      <div className="chat-main">
        {selectedConversation ? (
          <>
            <div className="chat-header">
              <div className="chat-user-info">
                <div className="chat-avatar">
                  <FiUser />
                </div>
                <div>
                  <h3>{selectedConversation.other_user.name}</h3>
                  <span className="user-role">{selectedConversation.other_user.role}</span>
                </div>
              </div>
            </div>

            <div className="chat-messages">
              {messages.length === 0 ? (
                <div className="no-messages">
                  <FiMessageCircle className="no-msg-icon" />
                  <p>No messages yet</p>
                  <p className="no-msg-hint">Start the conversation!</p>
                </div>
              ) : (
                messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`message ${msg.sender_id === user.id ? 'sent' : 'received'}`}
                  >
                    <div className="message-bubble">
                      {msg.image_path && (
                        <img 
                          src={msg.image_path} 
                          alt="Attachment" 
                          className="message-image"
                        />
                      )}
                      {msg.message_text && <p>{msg.message_text}</p>}
                      <span className="message-time">
                        {new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            <form className="chat-input-form" onSubmit={handleSendMessage}>
              {imagePreview && (
                <div className="image-preview-container">
                  <img src={imagePreview} alt="Preview" className="image-preview" />
                  <button type="button" className="btn-remove-image" onClick={clearImage} title="Remove image">
                    ×
                  </button>
                </div>
              )}
              <div className="input-row">
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="image/*"
                  onChange={handleImageSelect}
                  style={{ display: 'none' }}
                />
                <button 
                  type="button" 
                  className="btn-attach-image"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                >
                  <FiImage />
                </button>
                <input
                  type="text"
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  placeholder="Type a message..."
                  disabled={loading}
                />
                <button type="submit" disabled={loading || (!newMessage.trim() && !selectedImage)}>
                  <FiSend />
                </button>
              </div>
            </form>
          </>
        ) : (
          <div className="no-conversation-selected">
            <FiMessageCircle className="no-conv-icon" />
            <p>Select a conversation to start chatting</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatWindow;
