import { Route, Routes } from "react-router";
import "./App.css";
import Sidebar from "./components/sidebar/Sidebar";
import Home from "./pages/Home";
import Insights from "./pages/Insights";
import Upload from "./pages/Upload";

function App() {
  return (
    <div className="h-screen flex">
      <Sidebar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/insights" element={<Insights />} />
      </Routes>
    </div>
  );
}

export default App;
