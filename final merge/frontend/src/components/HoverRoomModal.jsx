import React, { useEffect, useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { roomLog, batteryPercentage } from '../assets/sampleRoomData';
import NotiMenu from './modalMenus/NotiMenu';
import TimePickerMenu from './modalMenus/TimePickerMenu';

    const BatteryIcon = ({ percentage }) => (
        <svg width="22" height="10" viewBox="0 0 40 20" style={{ position: 'absolute', top: '15', right: '10' }}>
            <rect x="1" y="1" width="34" height="18" rx="3" ry="3" fill="none" stroke="white" strokeWidth="2" />
            <rect x="35" y="6" width="4" height="8" rx="2" ry="2" fill="white" />
            <rect x="3" y="3" width={percentage * 0.3} height="14" fill={percentage > 20 ? "#4091F7" : "red"} />
        </svg>
    );


const HoverRoomModal = ({ room, style }) => {

    const getHoursStorageKey = () => {
        return `room_hours_${room.name.toLowerCase().replace(/\s+/g, '_')}`;
    };

    const hours =localStorage.getItem(getHoursStorageKey());

    const [opening, closing] = hours? hours.split(' - ') : [null, null];

    const [isHovered, setIsHovered] = useState(false);

    // Function to handle when modal is hovered
    const handleMouseEnter = () => {
        setIsHovered(true);
    };

    const handleMouseLeave = () => {
        setTimeout(() => {
            setIsHovered(false);
        }, 200); // Adjust delay as needed
    };

    const hoverModalStyle = {
        transform: 'translate(-50%, -50%)',
        backgroundColor: '#333333',
        border: `2px solid ${room.color || '#4091F7'}`,
        borderRadius: '8px',
        padding: '10px',
        zIndex: 1000,
        maxWidth: '200px', // Reduced width
        width: '100%',
        fontFamily: '"Roboto", sans-serif',
        fontWeight: 300,
        fontStyle: 'normal',
        boxShadow: '12px 7px 7px'
    };
    

    const headerStyle = {
        color: '#FFFFFF',
        marginTop: '0',
        marginBottom: '10px', // Reduced margin
        fontSize: '15px' // Smaller font
    };

    const hoursInfoStyle = {
        color: '#FFFFFF',
        fontSize: '10px' // Smaller font
    };

    return (
        <div id = 'hover-modal'
            style={{ ...hoverModalStyle, ...style }}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
        >
            <h2 style={headerStyle}>{room.name}</h2>
            <BatteryIcon percentage={batteryPercentage} />
            <p style={hoursInfoStyle}>
                <strong>Hours:</strong> 
                {room.openingTime || '??:??'} - {room.closingTime || '??:??'}
            </p>
        </div>
    );
};

export default HoverRoomModal;
