import React, { useEffect, useRef, useState } from 'react';
import TimePicker from 'react-time-picker';
import { ChevronLeft } from 'lucide-react';

const TimePickerMenu = ({ isOpen, onClose, onTimeSet, type = 'opening', currentStep = 1, onStepChange, onTypeChange }) => {
    const [selectedTime, setSelectedTime] = useState(null);

    const menuStyle = {
        position: 'absolute',
        top: '-182px',
        left: '27%',
        transform: 'translateX(-50%)',
        backgroundColor: '#333333',
        border: '1px solid #4091F7',
        borderRadius: '4px',
        padding: '12px',
        zIndex: 1001,
        width: '200px'
    };
    
    const headerStyle = {
        color: '#FFFFFF',
        margin: '0 0 12px 0',
        fontSize: '16px',
        fontWeight: '500',
        textAlign: 'center',
        position: 'relative'
    };
    
    const buttonStyle = {
        backgroundColor: '#4091F7',
        color: '#FFFFFF',
        border: 'none',
        padding: '8px 16px',
        borderRadius: '4px',
        cursor: 'pointer',
        width: '100%',
        marginTop: '12px'
    };
    
    const cancelButtonStyle = {
        ...buttonStyle,
        backgroundColor: 'transparent',
        border: '1px solid #4091F7',
        marginTop: '8px'
    };

    const backButtonStyle = {
        position: 'absolute',
        left: 0,
        top: '50%',
        transform: 'translateY(-50%)',
        background: 'none',
        border: 'none',
        color: '#FFFFFF',
        cursor: 'pointer',
        padding: 0,
        display: 'flex',
        alignItems: 'center'
    };

    const handleNextStep = () => {
        if (selectedTime) {
            onTimeSet(selectedTime, type);
            if (currentStep === 1) {
                onStepChange(2);
                onTypeChange(type === 'opening' ? 'closing' : 'opening');
            } else {
                onClose();
            }
            setSelectedTime(null)
        }
    };

    const handleBack = () => {
        onStepChange(1);
        onTypeChange(type === 'opening' ? 'closing' : 'opening');
        setSelectedTime(null)
    };

    if (!isOpen) return null;

    return (
        <div style={menuStyle}>
            <div style={headerStyle}>
                {currentStep === 2 && (
                    <button style={backButtonStyle} onClick={handleBack}>
                        <ChevronLeft size={20} />
                    </button>
                )}
                {type === 'opening' ? 'Opening Time' : 'Closing Time'}
            </div>
            
            <TimePicker
                onChange={setSelectedTime}
                value={selectedTime}
                clockIcon={null}
                clearIcon={null}
                className="time-picker"
            />
            
            <button 
                style={buttonStyle} 
                onClick={handleNextStep}
                disabled={!selectedTime}
            >
                {currentStep === 2 ? 'Set Hours' : `Set ${type === 'opening' ? 'Opening' : 'Closing'} Time`}
            </button>
            
            <button 
                style={cancelButtonStyle}
                onClick={onClose}
            >
                Cancel
            </button>
        </div>
    );
};

export default TimePickerMenu;