import { Route, Routes } from "react-router";
import "./App.css";
import Sidebar from "./components/sidebar/Sidebar";
import Home from "./pages/Home";
import Upload from "./pages/Upload";

function App() {
  return (
    <div className="h-screen flex">
      <Sidebar className="fixed" />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<Upload />} />
      </Routes>
    </div>
  );
}

export default App;
