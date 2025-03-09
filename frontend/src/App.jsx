import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import Home from "./pages/Home";
import Navbar from "./Components/Navbar";

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AddItemPage from "./pages/AddItemPage";
import BillItemPage from "./pages/BillItemPage";
import BillListPage from "./pages/BillListPage";
import PredictPage from "./pages/PredictPage";

function App() {
  return (
    <div className="App">
      <Router>
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/add-item" element={<AddItemPage />} />
            <Route path="/bill-item" element={<BillListPage />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/bill/:part_id" element={<BillItemPage />} />
          </Routes>
        </main>
      </Router>
    </div>
  );
}

export default App;
