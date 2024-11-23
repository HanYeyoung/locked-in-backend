import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Header from './Header';
import Floors from './Floors';
import Buildings from './Buildings';
const App = () => {
  return (
    <BrowserRouter>
      <Header />
      <Routes>
        <Route path="/" element={<Navigate to="/buildings" replace />} />
        <Route path="/buildings" element={<Buildings />} />
        <Route path="/buildings/:buildingId/floors" element={<Floors />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
