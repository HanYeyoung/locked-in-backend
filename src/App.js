import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Header from "./Header";
import Floors from "./Floors";
import Buildings from "./Buildings";
import FloorInterface from "./FloorInterface";
const App = () => {
    return (
        <BrowserRouter>
            <Header />
            <Routes>
                <Route
                    path="/"
                    element={<Navigate to="/buildings" replace />}
                />
                <Route path="/buildings" element={<Buildings />} />
                <Route
                    path="/buildings/:buildingId/floors"
                    element={<Floors />}
                />
                <Route
                    path="/buildings/:buildingId/floors/:floorId"
                    element={<FloorInterface />}
                />
            </Routes>
        </BrowserRouter>
    );
};

export default App;
