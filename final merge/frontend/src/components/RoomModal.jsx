import React, { useEffect, useState, useRef } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { roomLog, batteryPercentage } from '../assets/sampleRoomData';
import NotiMenu from './modalMenus/NotiMenu';
import TimePickerMenu from './modalMenus/TimePickerMenu';
import { motion, AnimatePresence } from 'framer-motion';

const RoomModal = ({ room, onClose }) => {
    const [isLogOpen, setIsLogOpen] = useState(false);
    const [alertStatus, setAlertStatus] = useState("off")
    const [isNotiMenuOpen, setIsNotiMenuOpen] = useState(false);
    const [isTimePickerOpen, setIsTimePickerOpen] = useState(false);
    const [timePickerType, setTimePickerType] = useState('opening');
    const [timePickerStep, setTimePickerStep] = useState(1);
    const modalRef = useRef(null);
    const hoursRef = useRef({
        opening: null,
        closing: null
    });
    const [, forceUpdate] = useState({})

    useEffect(() => {
        const savedHours = localStorage.getItem(getHoursStorageKey());
        const savedAlertStatus = localStorage.getItem(getAlertStorageKey());
        
        if (savedHours) {
            const [opening, closing] = savedHours.split(' - ');
            if (opening !== '??:??') hoursRef.current.opening = opening;
            if (closing !== '??:??') hoursRef.current.closing = closing;
            forceUpdate({});
        }

        if (savedAlertStatus) {
            setAlertStatus(savedAlertStatus);
        }
    }, [room]);

    const modalContainerStyle = {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        pointerEvents: 'none',
    };

    const modalStyle = {
        backgroundColor: '#333333',
        border: `2px solid ${room.color || '#4091F7'}`,
        borderRadius: '8px',
        padding: '20px',
        maxWidth: '500px',
        width: '100%',
        fontFamily: '"Roboto", sans-serif',
        fontWeight: 300,
        fontStyle: 'normal',
        pointerEvents: 'auto',
        position: 'relative'
    };

    const headerStyle = {
        color: '#FFFFFF',
        marginTop: '0',
        marginBottom: '20px',
        borderBottom: `1px solid ${room.color || '#4091F7'}`,
        paddingBottom: '10px',
        position: 'relative',
        fontSize: '24px'
    };

    const hoursInfoStyle = {
        color: '#FFFFFF',
        marginBottom: '10px',
        fontSize: '16px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
    };

    const timePickerStyle = {
        position: 'fixed',
        top: 0,
        left: 0,
        bottom: 0,
        right: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        pointerEvents: 'none',
        // transform: 'translateY(-100%)', // Move up by its own height
        zIndex: 1001
    };

    const logInfoStyle = {
        color: '#FFFFFF',
        marginBottom: '10px',
        fontSize: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
    };

    const logLineStyle = {
        color: '#FFFFFF',
        marginBottom: '10px',
        marginRight: '10px',
        fontSize: '16px',
        fontStyle: 'normal'
    };

    const closeButtonStyle = {
        backgroundColor: 'transparent', 
        color: '#FFF',             
        border: `2px solid ${room.color || '#4091F7'}`,   
        padding: '10px 20px',          
        borderRadius: '4px',          
        cursor: 'pointer',            
        float: 'right',      
    };
    

    const dropdownStyle = {
        backgroundColor: '#444444',
        border: '1px solid #4091F7',
        borderRadius: '4px',
        maxHeight: '200px',
        overflowY: 'auto',
    };

    const dropdownItemStyle = {
        padding: '10px',
        borderBottom: '1px solid #555555',
    };

    const toggleLogStyle = {
        backgroundColor: 'transparent',
        border: 'none',
        color: '#FFFFFF',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        padding: 0,
    };

    // Generate a unique key for localStorage
    const getHoursStorageKey = () => {
        return `room_hours_${room.name.toLowerCase().replace(/\s+/g, '_')}`;
    };

    // Generate a unique key for localStorage
    const getAlertStorageKey = () => {
        return `alert_status_${room.name.toLowerCase().replace(/\s+/g, '_')}`;
    };

    const BatteryIcon = ({ percentage }) => (
        <svg width="26" height="14" viewBox="0 0 40 20" style={{ position: 'absolute', top: '0', right: '0' }}>
            <rect x="1" y="1" width="34" height="18" rx="3" ry="3" fill="none" stroke="white" strokeWidth="2" />
            <rect x="35" y="6" width="4" height="8" rx="2" ry="2" fill="white" />
            <rect x="3" y="3" width={percentage * 0.3} height="14" fill={percentage > 20 ? "#4091F7" : "red"} />
        </svg>
    );

    const sortedLog = [...roomLog].sort((a, b) => new Date(`${b.date} ${b.time}`) - new Date(`${a.date} ${a.time}`));

    const mostRecent =  `${roomLog[0].name} (${roomLog[0].hierarchy}) - ${roomLog[0].date} ${roomLog[0].time}`
    
    const handleTimeSet = (time, type) => {
        if (type === 'opening') {
            hoursRef.current.opening = time;
        } else {
            hoursRef.current.closing = time;
        }
        const hours = `${hoursRef.current.opening || '??:??'} - ${hoursRef.current.closing || '??:??'}`;
        localStorage.setItem(getHoursStorageKey(), hours);
        forceUpdate({});
    };

    const handleNotiChange = (status) => {
        setAlertStatus(status);
        localStorage.setItem(getAlertStorageKey(), status);
        setIsNotiMenuOpen(false);
    };

    const modalVariants = {
        initial: {
            scale: 0.3,
            opacity: 0
        },
        animate: {
            scale: 1,
            opacity: 1,
            transition: {
                duration: 0.3,
                ease: [0.4, 0, 0.2, 1]
            }
        },
        exit: {
            opacity: 0,
            scale: 0.3,
            transition: {
                duration: 0.3,
            }
        }
    };

    // <div style={modalStyle}>

    return <>
        <div style={modalContainerStyle}>
            <motion.div
                ref={modalRef}
                initial="initial"
                animate="animate"
                exit="exit"
                variants={modalVariants}
                style={modalStyle}
            >
                <h2 style={headerStyle}>
                    {room.name}
                    <div 
                        className="bell-icon"
                        onClick={() => setIsNotiMenuOpen(!isNotiMenuOpen)}
                    >
                        <NotiMenu 
                            isOpen={isNotiMenuOpen}
                            onStatusChange={handleNotiChange}
                            currentStatus={alertStatus}
                        />
                    </div>
                    <BatteryIcon percentage={batteryPercentage} />
                </h2>
                <p style={hoursInfoStyle}>
                    <strong>Hours:</strong> 
                    <span 
                        onClick={() => {
                            setTimePickerType('opening');
                            setIsTimePickerOpen(true);
                            setTimePickerStep(1); 
                        }}
                        style={{ cursor: 'pointer', color: '#FFFFFF' }}
                    >
                        {hoursRef.current.opening || '??:??'}
                    </span>
                    {' - '}
                    <span
                        onClick={() => {
                            setTimePickerType('closing');
                            setIsTimePickerOpen(true);
                            setTimePickerStep(1); 
                        }}
                        style={{ cursor: 'pointer', color: '#FFFFFF' }}
                    >
                        {hoursRef.current.closing || '??:??'}
                    </span>
                </p>
                <div style={logInfoStyle}>
                    <button onClick={() => setIsLogOpen(!isLogOpen)} style={toggleLogStyle}>
                        <p style={logLineStyle}><strong>Entry Log: </strong>{mostRecent}</p>
                        {isLogOpen ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                    </button>
                    {isLogOpen && (
                        <div style={dropdownStyle}>
                            {sortedLog.map((entry, index) => (
                                <div key={index} style={dropdownItemStyle}>
                                    {entry.name} ({entry.hierarchy}) - {entry.date} {entry.time}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
                <button style={closeButtonStyle} onClick={onClose}>Close</button>

                {isTimePickerOpen && (
                    <TimePickerMenu
                        isOpen={isTimePickerOpen}
                        onClose={() => {
                            setIsTimePickerOpen(false);
                            setTimePickerStep(1); 
                        }}
                        onTimeSet={handleTimeSet}
                        onStepChange={setTimePickerStep} 
                        type={timePickerType}
                        currentStep={timePickerStep}
                        onTypeChange={setTimePickerType}  
                        style={timePickerStyle}              
                    />
                )}
            </motion.div>
        </div>
    </>;
};

export default RoomModal;