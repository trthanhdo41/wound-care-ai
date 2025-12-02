import React, { useState, useRef, useEffect } from 'react';
import { FiX, FiSend } from 'react-icons/fi';
import { FaHeartbeat } from 'react-icons/fa';
import ReactMarkdown from 'react-markdown';
import './MedicalChatBot.css';

export const MedicalChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { 
      text: "Xin chào! Tôi là trợ lý y tế AI của WoundCare.\n\nTôi có thể giúp bạn về:\n• Chăm sóc vết loét bàn chân tiểu đường\n• Hiểu kết quả phân tích AI\n• Hướng dẫn chăm sóc vết thương\n• Tư vấn y tế cơ bản\n\nBạn cần hỗ trợ gì?", 
      isBot: true,
      timestamp: new Date()
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;
    
    const userMessage = input;
    const userMsg = {
      text: userMessage,
      isBot: false,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const systemPrompt = `Bạn là trợ lý y tế AI chuyên nghiệp của WoundCare - hệ thống phân tích vết loét bàn chân tiểu đường bằng AI.

THÔNG TIN HỆ THỐNG:
- Chức năng: Phân tích hình ảnh vết loét bằng AI, đánh giá mức độ nguy hiểm, theo dõi quá trình lành
- Công nghệ: SegFormer AI model, phân tích màu sắc, kích thước, độ nhám bề mặt
- Kết quả: Mức độ rủi ro (Low/Medium/High/Critical), kích thước vết thương, phân tích màu sắc
- Hỗ trợ: Tư vấn chăm sóc vết thương, giải thích kết quả AI, hướng dẫn điều trị

KIẾN THỨC Y TẾ:
- Vết loét tiểu đường: Biến chứng nghiêm trọng, cần theo dõi thường xuyên
- Màu sắc vết thương:
  + Đỏ/Hồng: Mô hạt tốt, đang lành
  + Vàng: Mô hoại tử ướt, cần làm sạch
  + Đen/Nâu: Mô hoại tử khô, nguy hiểm
- Chăm sóc: Giữ sạch, băng bó đúng cách, kiểm soát đường huyết, khám bác sĩ định kỳ
- Dấu hiệu nguy hiểm: Sưng đỏ, mủ, mùi hôi, sốt, đau tăng → Cần gặp bác sĩ ngay

PHONG CÁCH TRẢ LỜI:
- Chuyên nghiệp, thân thiện, dễ hiểu
- Dùng ngôn ngữ y tế nhưng giải thích đơn giản
- Luôn khuyến khích gặp bác sĩ khi cần thiết
- Không chẩn đoán hoặc kê đơn thuốc
- Trả lời ngắn gọn, súc tích (2-4 câu)
- Dùng emoji y tế vừa phải: 🏥, 💊, 🩺, ⚕️, ✅, ⚠️`;

      // Build Gemini contents array with proper format
      const geminiContents = [];
      
      // Add system prompt as first user message
      geminiContents.push({
        role: 'user',
        parts: [{ text: systemPrompt }]
      });
      
      geminiContents.push({
        role: 'model',
        parts: [{ text: 'Tôi hiểu. Tôi là trợ lý y tế AI của WoundCare, chuyên về vết loét bàn chân tiểu đường. Tôi sẽ trả lời chuyên nghiệp, thân thiện và ngắn gọn.' }]
      });
      
      // Add conversation history
      messages.filter(msg => !msg.isTyping).forEach(msg => {
        geminiContents.push({
          role: msg.isBot ? 'model' : 'user',
          parts: [{ text: msg.text }]
        });
      });
      
      // Add current user message
      geminiContents.push({
        role: 'user',
        parts: [{ text: userMessage }]
      });
      
      // Convert Gemini format to Groq format
      const groqMessages = [
        { role: 'system', content: systemPrompt }
      ];
      
      // Add conversation history
      messages.filter(msg => !msg.isTyping).forEach(msg => {
        groqMessages.push({
          role: msg.isBot ? 'assistant' : 'user',
          content: msg.text
        });
      });
      
      // Add current user message
      groqMessages.push({
        role: 'user',
        content: userMessage
      });
      
      const response = await fetch(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_GROQ_API_KEY'
          },
          body: JSON.stringify({
            model: 'llama-3.3-70b-versatile',
            messages: groqMessages,
            temperature: 0.7,
            max_tokens: 300,
          })
        }
      );

      const data = await response.json();
      
      console.log('Groq Response:', data); // Debug log
      
      if (!response.ok) {
        console.error('API Error:', data);
        throw new Error(`API Error: ${response.status}`);
      }

      setIsTyping(false);
      
      // Extract response from Groq format
      let botResponse = '';
      
      if (data.choices && data.choices.length > 0) {
        botResponse = data.choices[0].message.content;
      } else if (data.error) {
        console.error('Groq Error:', data.error);
        botResponse = 'Xin lỗi, có lỗi xảy ra. Vui lòng thử lại. 🏥';
      } else {
        botResponse = 'Xin lỗi, tôi không thể trả lời lúc này. 🏥';
      }
      
      // Typing effect
      if (botResponse) {
        const fullText = botResponse;
        let currentText = '';
        
        for (let i = 0; i < fullText.length; i++) {
          setTimeout(() => {
            currentText += fullText[i];
            const isLastChar = i === fullText.length - 1;
            
            setMessages(prev => {
              const withoutTyping = prev.filter(msg => !msg.isTyping);
              return [...withoutTyping, { 
                text: currentText, 
                isBot: true, 
                timestamp: new Date(),
                isTyping: !isLastChar 
              }];
            });
          }, i * 20);
        }
      }
      
    } catch (error) {
      console.error('Chat Error:', error);
      setIsTyping(false);
      const errorMsg = {
        text: 'Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau hoặc liên hệ bác sĩ. 🏥',
        isBot: true,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMsg]);
    }
  };

  const quickReplies = [
    "Cách chăm sóc vết loét",
    "Giải thích kết quả AI",
    "Dấu hiệu nguy hiểm",
    "Kiểm soát đường huyết"
  ];

  return (
    <>
      {/* Chat Button */}
      <div className="medical-chat-button-container">
        <div className="medical-chat-button-wrapper">
          {!isOpen && (
            <span className="medical-chat-pulse">
              <span className="medical-chat-pulse-ring"></span>
              <span className="medical-chat-pulse-dot"></span>
            </span>
          )}
          <button
            className="medical-chat-button"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <FiX size={24} /> : <FaHeartbeat size={24} />}
          </button>
        </div>
      </div>

      {/* Chat Widget */}
      {isOpen && (
        <div className="medical-chat-widget">
          {/* Header */}
          <div className="medical-chat-header">
            <div className="medical-chat-header-content">
              <div className="medical-chat-avatar-wrapper">
                <div className="medical-chat-avatar">
                  <FaHeartbeat size={24} />
                </div>
                <span className="medical-chat-status"></span>
              </div>
              <div className="medical-chat-header-info">
                <h3>WoundCare Assistant</h3>
                <p>
                  <span className="medical-chat-online-dot"></span>
                  Trả lời ngay lập tức
                </p>
              </div>
            </div>
          </div>

          {/* Messages */}
          <div className="medical-chat-messages">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`medical-chat-message ${msg.isBot ? 'bot' : 'user'}`}
              >
                <div className="medical-chat-message-bubble">
                  {msg.isBot ? (
                    <ReactMarkdown
                      components={{
                        p: ({node, ...props}) => <span {...props} />,
                        strong: ({node, ...props}) => <strong style={{fontWeight: 700, color: '#0ea5e9'}} {...props} />,
                        em: ({node, ...props}) => <em style={{fontStyle: 'italic', color: '#059669'}} {...props} />,
                        ul: ({node, ...props}) => <ul style={{marginLeft: '20px', marginTop: '8px'}} {...props} />,
                        ol: ({node, ...props}) => <ol style={{marginLeft: '20px', marginTop: '8px'}} {...props} />,
                        li: ({node, ...props}) => <li style={{marginBottom: '4px'}} {...props} />
                      }}
                    >
                      {msg.text}
                    </ReactMarkdown>
                  ) : (
                    <p>{msg.text}</p>
                  )}
                </div>
                <p className="medical-chat-message-time">
                  {msg.timestamp.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            ))}
            
            {isTyping && (
              <div className="medical-chat-message bot">
                <div className="medical-chat-message-bubble">
                  <div className="medical-chat-typing">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Replies */}
          {messages.length === 1 && (
            <div className="medical-chat-quick-replies">
              {quickReplies.map((reply, i) => (
                <button
                  key={i}
                  className="medical-chat-quick-reply"
                  onClick={() => {
                    setInput(reply);
                    setTimeout(() => handleSend(), 0);
                  }}
                >
                  {reply}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div className="medical-chat-input-container">
            <div className="medical-chat-input-wrapper">
              <input
                type="text"
                placeholder="Nhập câu hỏi của bạn..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                className="medical-chat-input"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isTyping}
                className="medical-chat-send-button"
              >
                <FiSend size={18} />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
