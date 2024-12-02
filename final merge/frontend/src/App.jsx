import React, {useState} from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import MapboxContainer from './components/MapboxContainer';
import ConferenceRooms from './components/ConferenceRooms'
import NewMapPage from './pages/NewMapPage';
import Navigation from './components/Navigation';
import LoginPage from './components/LoginPage';
import './App.css';

function App() {
  // State to track if the user is logged in
  const [loggedIn, setLoggedIn] = useState(false);
  const [username, setUsername] = useState('')

    // Function to handle successful login
    const handleLogin = (username, password) => {
    // For simplicity, we're assuming login is successful here
    // In a real application, you'd validate login credentials with an API
    console.log('Logged in:', { username, password });
    
    // You could validate here before setting loggedIn to true
    if ((username === 'admin' || username === 'joeuntrecht' || username === 'eligauger') && password === 'password') {
      setUsername(username)
      setLoggedIn(true);
    } else {
      alert("Invalid login credentials. Please try again.");
    }
  };

  //what does <Navigation /> do?
  return (
    <Router>
      <Navigation/>
      <Routes>
        {loggedIn ? <Route path="/" element={<MapboxContainer username={username} />}/>: <Route path="/" element= {<LoginPage onLogin={handleLogin} />}/>}
        <Route path="/new-map-page" element={<NewMapPage />} />
        <Route path="/conference-rooms" element={<ConferenceRooms username={username} />} />      
      </Routes>
    </Router>
  );
}

export default App;
