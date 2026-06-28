import { BrowserRouter, Routes, Route } from "react-router-dom"
import TopBar from "./components/TopBar"
import FloatingPets from "./components/FloatingPets"
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
      <FloatingPets />
      <TopBar />
      <main className="mx-auto max-w-[1400px] px-4 md:px-6 py-8">
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
      <footer className="mx-auto max-w-[1400px] px-6 py-8 text-center text-xs text-ink/40">
        🐾 Petopia Intelligence Hub · Chronos + N-HiTS + CatBoost ensemble · MCP agent
      </footer>
    </BrowserRouter>
  )
}
