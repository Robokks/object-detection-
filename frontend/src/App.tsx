import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import "./App.css";
import DatasetPage from "./pages/DatasetPage";
import TrainPage from "./pages/TrainPage";
import DetectPage from "./pages/DetectPage";

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Vision Object Detection Studio</h1>
        <nav>
          <NavLink to="/dataset" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            1. Dataset
          </NavLink>
          <NavLink to="/train" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            2. Train
          </NavLink>
          <NavLink to="/detect" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
            3. Detect
          </NavLink>
        </nav>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Navigate to="/dataset" replace />} />
          <Route path="/dataset" element={<DatasetPage />} />
          <Route path="/train" element={<TrainPage />} />
          <Route path="/detect" element={<DetectPage />} />
        </Routes>
      </main>
    </div>
  );
}
