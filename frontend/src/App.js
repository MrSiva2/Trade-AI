import { useState, useEffect, useCallback } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router-dom";
import axios from "axios";
import { Toaster, toast } from "sonner";
import {
  LayoutDashboard,
  Database,
  Brain,
  Activity,
  LineChart,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
  Upload,
  FolderOpen,
  Play,
  Square,
  Download,
  RefreshCw,
  Zap,
  TrendingUp,
  TrendingDown,
  Terminal
} from "lucide-react";

import Dashboard from "./pages/Dashboard";
import DataManagement from "./pages/DataManagement";
import ModelManagement from "./pages/ModelManagement";
import Training from "./pages/Training";
import Backtesting from "./pages/Backtesting";
import Logs from "./pages/Logs";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

// Sidebar Component
const Sidebar = ({ collapsed, setCollapsed }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { icon: LayoutDashboard, label: "Dashboard", path: "/" },
    { icon: Database, label: "Data", path: "/data" },
    { icon: Brain, label: "Models", path: "/models" },
    { icon: Activity, label: "Training", path: "/training" },
    { icon: LineChart, label: "Backtest", path: "/backtest" },
    { icon: Terminal, label: "Logs", path: "/logs" },
  ];

  return (
    <div className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''}`} data-testid="sidebar">
      <div className="flex flex-col h-full">
        {/* Logo */}
        <div className="p-4 border-b border-border flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-primary flex items-center justify-center">
            <Zap className="w-5 h-5 text-primary-foreground" />
          </div>
          {!collapsed && (
            <span className="font-bold text-lg tracking-tight">TradeAI</span>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className={`nav-item w-full ${isActive ? 'nav-item-active' : ''}`}
                data-testid={`nav-${item.label.toLowerCase()}`}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                {!collapsed && <span>{item.label}</span>}
              </button>
            );
          })}
        </nav>

        {/* Collapse button */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-4 border-t border-border flex items-center justify-center hover:bg-accent transition-colors"
          data-testid="sidebar-toggle"
        >
          {collapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <ChevronLeft className="w-5 h-5" />
          )}
        </button>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-background noise-overlay">
      <BrowserRouter>
        <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
        <main className={`main-content ${collapsed ? 'main-content-collapsed' : ''}`}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/data" element={<DataManagement />} />
            <Route path="/models" element={<ModelManagement />} />
            <Route path="/training" element={<Training />} />
            <Route path="/backtest" element={<Backtesting />} />
            <Route path="/logs" element={<Logs />} />
          </Routes>
        </main>
        <Toaster 
          theme="dark" 
          position="bottom-right"
          toastOptions={{
            style: {
              background: 'hsl(240 6% 9%)',
              border: '1px solid hsl(240 4% 20%)',
              color: 'hsl(0 0% 98%)'
            }
          }}
        />
      </BrowserRouter>
    </div>
  );
}

export default App;
