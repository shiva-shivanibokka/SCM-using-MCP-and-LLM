import { BrowserRouter, Routes, Route } from "react-router-dom"
import Sidebar from "./components/Sidebar"
import Executive from "./pages/Executive"
import Inventory from "./pages/Inventory"
import Forecast from "./pages/Forecast"
import Suppliers from "./pages/Suppliers"
import Stores from "./pages/Stores"
import Analytics from "./pages/Analytics"
import AIAssistant from "./pages/AIAssistant"
import MLOps from "./pages/MLOps"

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6 min-h-screen bg-cream">
          <Routes>
            <Route path="/" element={<Executive />} />
            <Route path="/inventory" element={<Inventory />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/suppliers" element={<Suppliers />} />
            <Route path="/stores" element={<Stores />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/assistant" element={<AIAssistant />} />
            <Route path="/mlops" element={<MLOps />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
