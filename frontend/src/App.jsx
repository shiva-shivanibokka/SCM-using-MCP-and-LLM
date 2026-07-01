import { BrowserRouter, Routes, Route } from "react-router-dom"
import TopBar from "./components/TopBar"
import FloatingPets from "./components/FloatingPets"
import Executive from "./pages/Executive"
import Inventory from "./pages/Inventory"
import Forecast from "./pages/Forecast"
import Suppliers from "./pages/Suppliers"
import Stores from "./pages/Stores"
import Analytics from "./pages/Analytics"
import Recommendations from "./pages/Recommendations"
import Stockout from "./pages/Stockout"
import Anomaly from "./pages/Anomaly"
import WhatIf from "./pages/WhatIf"
import AIAssistant from "./pages/AIAssistant"
import MLOps from "./pages/MLOps"

export default function App() {
  return (
    <BrowserRouter>
      <FloatingPets />
      <TopBar />
      <main className="mx-auto max-w-[1840px] px-5 md:px-10 py-8">
        <Routes>
          <Route path="/" element={<Executive />} />
          <Route path="/inventory" element={<Inventory />} />
          <Route path="/forecast" element={<Forecast />} />
          <Route path="/suppliers" element={<Suppliers />} />
          <Route path="/stores" element={<Stores />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/recommendations" element={<Recommendations />} />
          <Route path="/stockout" element={<Stockout />} />
          <Route path="/anomaly" element={<Anomaly />} />
          <Route path="/whatif" element={<WhatIf />} />
          <Route path="/assistant" element={<AIAssistant />} />
          <Route path="/mlops" element={<MLOps />} />
        </Routes>
      </main>
      <footer className="px-10 py-6 text-center text-base text-ink/45 border-t border-ink/5">
        🐾 Petopia Intelligence Hub · Chronos + N-HiTS + CatBoost ensemble · MCP agent
      </footer>
    </BrowserRouter>
  )
}
