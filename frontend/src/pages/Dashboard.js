import { useState, useEffect } from "react";
import axios from "axios";
import { API } from "../App";
import {
  Brain,
  Activity,
  TrendingUp,
  BarChart3,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  ArrowUpRight,
  ArrowDownRight
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Progress } from "../components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar
} from "recharts";

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [modelPerformance, setModelPerformance] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [statsRes, sessionsRes, performanceRes] = await Promise.all([
        axios.get(`${API}/dashboard/stats`),
        axios.get(`${API}/training/sessions`),
        axios.get(`${API}/dashboard/model-performance`)
      ]);
      
      setStats(statsRes.data);
      setSessions(sessionsRes.data.sessions || []);
      setModelPerformance(performanceRes.data.performance || []);
      
      // Auto-select first session (running or most recent)
      if (!selectedSession && sessionsRes.data.sessions?.length > 0) {
        setSelectedSession(sessionsRes.data.sessions[0]);
      } else if (selectedSession) {
        // Update selected session with latest data
        const updated = sessionsRes.data.sessions?.find(s => s.id === selectedSession.id);
        if (updated) {
          setSelectedSession(updated);
        }
      }
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
    } finally {
      setLoading(false);
    }
  };

  const StatCard = ({ title, value, icon: Icon, change, changeType, subtitle }) => (
    <Card className="stat-card card-hover" data-testid={`stat-${title.toLowerCase().replace(/\s/g, '-')}`}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div>
            <p className="metric-label mb-1">{title}</p>
            <p className="metric-value">{value}</p>
            {subtitle && (
              <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
            )}
          </div>
          <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center">
            <Icon className="w-5 h-5 text-primary" />
          </div>
        </div>
        {change !== undefined && (
          <div className="flex items-center gap-1 mt-3">
            {changeType === "positive" ? (
              <ArrowUpRight className="w-4 h-4 text-green-500" />
            ) : (
              <ArrowDownRight className="w-4 h-4 text-red-500" />
            )}
            <span className={`text-sm font-mono ${changeType === "positive" ? "text-green-500" : "text-red-500"}`}>
              {change}%
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]" data-testid="dashboard-loading">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  const mockTrainingData = Array.from({ length: 20 }, (_, i) => ({
    epoch: i + 1,
    loss: Math.max(0.1, 1 - i * 0.04 + Math.random() * 0.05),
    accuracy: Math.min(0.95, 0.5 + i * 0.02 + Math.random() * 0.03)
  }));

  return (
    <div className="space-y-6 animate-fade-in" data-testid="dashboard">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Trading AI Model Hub Overview
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Clock className="w-4 h-4" />
          <span className="font-mono">Last updated: {new Date().toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Models"
          value={stats?.total_models || 0}
          icon={Brain}
          subtitle={`${stats?.custom_models || 0} custom`}
        />
        <StatCard
          title="Training Sessions"
          value={stats?.training_sessions || 0}
          icon={Activity}
          subtitle={`${stats?.active_trainings || 0} active`}
        />
        <StatCard
          title="Backtests"
          value={stats?.backtest_count || 0}
          icon={BarChart3}
        />
        <StatCard
          title="Latest Return"
          value={stats?.latest_backtest?.total_return ? `${stats.latest_backtest.total_return.toFixed(2)}%` : "N/A"}
          icon={TrendingUp}
          change={stats?.latest_backtest?.total_return?.toFixed(2)}
          changeType={stats?.latest_backtest?.total_return > 0 ? "positive" : "negative"}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Training Progress Chart */}
        <Card className="panel" data-testid="training-chart-panel">
          <CardHeader className="panel-header">
            <div className="flex items-center justify-between w-full">
              <CardTitle className="panel-title">Training Progress</CardTitle>
              <div className="flex items-center gap-2">
                {selectedSession && (
                  <span className={`badge ${
                    selectedSession.status === "running" ? "status-running" :
                    selectedSession.status === "completed" ? "status-completed" :
                    selectedSession.status === "failed" ? "status-failed" : "status-pending"
                  }`}>
                    {selectedSession.status}
                  </span>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-4">
            {/* Training Session Selector */}
            {sessions.length > 0 && (
              <div className="mb-4">
                <Select
                  value={selectedSession?.id}
                  onValueChange={(id) => {
                    const session = sessions.find(s => s.id === id);
                    setSelectedSession(session);
                  }}
                >
                  <SelectTrigger className="w-full" data-testid="session-selector">
                    <SelectValue placeholder="Select training session" />
                  </SelectTrigger>
                  <SelectContent>
                    {sessions.map((session) => (
                      <SelectItem key={session.id} value={session.id}>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs">{session.model_id?.slice(0, 8)}</span>
                          <span className="text-xs">
                            {session.status === "running" && "🟢"}
                            {session.status === "completed" && "✓"}
                            {session.status === "failed" && "✗"}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {session.current_epoch}/{session.total_epochs} epochs
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={selectedSession?.metrics?.loss?.length > 0 
                  ? selectedSession.metrics.loss.map((loss, i) => ({
                      epoch: i + 1,
                      loss,
                      accuracy: selectedSession.metrics.accuracy?.[i] || 0
                    }))
                  : mockTrainingData
                }>
                  <defs>
                    <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="accGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(240 4% 20%)" />
                  <XAxis 
                    dataKey="epoch" 
                    stroke="hsl(240 5% 45%)" 
                    tick={{ fill: 'hsl(240 5% 65%)', fontSize: 11 }}
                  />
                  <YAxis 
                    stroke="hsl(240 5% 45%)" 
                    tick={{ fill: 'hsl(240 5% 65%)', fontSize: 11 }}
                  />
                  <Tooltip 
                    contentStyle={{
                      background: 'hsl(240 6% 9%)',
                      border: '1px solid hsl(240 4% 20%)',
                      borderRadius: '6px',
                      color: 'hsl(0 0% 98%)'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#6366f1" 
                    fill="url(#lossGradient)"
                    strokeWidth={2}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#22c55e" 
                    fill="url(#accGradient)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            {selectedSession && (
              <div className="mt-4">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-muted-foreground">Progress</span>
                  <span className="font-mono">
                    {selectedSession.current_epoch}/{selectedSession.total_epochs}
                  </span>
                </div>
                <Progress 
                  value={(selectedSession.current_epoch / selectedSession.total_epochs) * 100} 
                />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Latest Backtest Performance */}
        <Card className="panel" data-testid="backtest-chart-panel">
          <CardHeader className="panel-header">
            <CardTitle className="panel-title">Latest Backtest</CardTitle>
          </CardHeader>
          <CardContent className="p-4">
            {stats?.latest_backtest ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="metric-label">Total Return</p>
                    <p className={`metric-value text-lg ${stats.latest_backtest.total_return >= 0 ? 'trading-positive' : 'trading-negative'}`}>
                      {stats.latest_backtest.total_return.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="metric-label">Sharpe Ratio</p>
                    <p className="metric-value text-lg">
                      {stats.latest_backtest.sharpe_ratio.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="metric-label">Max Drawdown</p>
                    <p className="metric-value text-lg trading-negative">
                      -{stats.latest_backtest.max_drawdown.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="metric-label">Total Trades</p>
                    <p className="metric-value text-lg">
                      {stats.latest_backtest.total_trades}
                    </p>
                  </div>
                </div>
                <div className="pt-4 border-t border-border">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-500" />
                      <span className="text-sm">Winning: {stats.latest_backtest.winning_trades}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <XCircle className="w-4 h-4 text-red-500" />
                      <span className="text-sm">Losing: {stats.latest_backtest.losing_trades}</span>
                    </div>
                  </div>
                  <div className="mt-3">
                    <div className="h-2 bg-muted rounded-full overflow-hidden flex">
                      <div 
                        className="bg-green-500 h-full"
                        style={{ 
                          width: `${(stats.latest_backtest.winning_trades / Math.max(1, stats.latest_backtest.total_trades)) * 100}%` 
                        }}
                      />
                      <div 
                        className="bg-red-500 h-full"
                        style={{ 
                          width: `${(stats.latest_backtest.losing_trades / Math.max(1, stats.latest_backtest.total_trades)) * 100}%` 
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[200px] text-muted-foreground">
                <BarChart3 className="w-12 h-12 mb-2 opacity-50" />
                <p>No backtest results yet</p>
                <p className="text-sm">Run a backtest to see results here</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Model Performance Section */}
      {modelPerformance.length > 0 && (
        <Card className="panel" data-testid="model-performance-panel">
          <CardHeader className="panel-header">
            <CardTitle className="panel-title">Model Performance Comparison</CardTitle>
          </CardHeader>
          <CardContent className="p-4">
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modelPerformance}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(240 4% 20%)" />
                  <XAxis 
                    dataKey="model_id" 
                    stroke="hsl(240 5% 45%)" 
                    tick={{ fill: 'hsl(240 5% 65%)', fontSize: 10 }}
                    tickFormatter={(value) => value.slice(0, 8)}
                  />
                  <YAxis 
                    stroke="hsl(240 5% 45%)" 
                    tick={{ fill: 'hsl(240 5% 65%)', fontSize: 11 }}
                    label={{ value: 'Return (%)', angle: -90, position: 'insideLeft', fill: 'hsl(240 5% 65%)' }}
                  />
                  <Tooltip 
                    contentStyle={{
                      background: 'hsl(240 6% 9%)',
                      border: '1px solid hsl(240 4% 20%)',
                      borderRadius: '6px',
                      color: 'hsl(0 0% 98%)'
                    }}
                    formatter={(value, name) => {
                      if (name === 'avg_return') return [`${value}%`, 'Avg Return'];
                      if (name === 'max_return') return [`${value}%`, 'Max Return'];
                      if (name === 'min_return') return [`${value}%`, 'Min Return'];
                      return [value, name];
                    }}
                  />
                  <Bar dataKey="avg_return" fill="#6366f1" name="Avg Return" />
                  <Bar dataKey="max_return" fill="#22c55e" name="Max Return" />
                  <Bar dataKey="min_return" fill="#ef4444" name="Min Return" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
              {modelPerformance.slice(0, 4).map((perf) => (
                <div key={perf.model_id} className="text-center p-3 bg-accent rounded-md">
                  <p className="text-xs text-muted-foreground font-mono mb-1">
                    {perf.model_id.slice(0, 8)}
                  </p>
                  <p className={`text-lg font-bold ${perf.avg_return >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {perf.avg_return >= 0 ? '+' : ''}{perf.avg_return}%
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Win Rate: {perf.win_rate}%
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      <Card className="panel" data-testid="quick-actions-panel">
        <CardHeader className="panel-header">
          <CardTitle className="panel-title">Quick Actions</CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <button 
              onClick={() => window.location.href = '/data'}
              className="flex flex-col items-center gap-2 p-4 rounded-md bg-accent hover:bg-accent/80 transition-colors"
              data-testid="action-upload-data"
            >
              <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                <Activity className="w-5 h-5 text-primary" />
              </div>
              <span className="text-sm font-medium">Upload Data</span>
            </button>
            <button 
              onClick={() => window.location.href = '/models'}
              className="flex flex-col items-center gap-2 p-4 rounded-md bg-accent hover:bg-accent/80 transition-colors"
              data-testid="action-new-model"
            >
              <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                <Brain className="w-5 h-5 text-primary" />
              </div>
              <span className="text-sm font-medium">New Model</span>
            </button>
            <button 
              onClick={() => window.location.href = '/training'}
              className="flex flex-col items-center gap-2 p-4 rounded-md bg-accent hover:bg-accent/80 transition-colors"
              data-testid="action-start-training"
            >
              <div className="w-10 h-10 rounded-full bg-green-500/10 flex items-center justify-center">
                <Activity className="w-5 h-5 text-green-500" />
              </div>
              <span className="text-sm font-medium">Start Training</span>
            </button>
            <button 
              onClick={() => window.location.href = '/backtest'}
              className="flex flex-col items-center gap-2 p-4 rounded-md bg-accent hover:bg-accent/80 transition-colors"
              data-testid="action-run-backtest"
            >
              <div className="w-10 h-10 rounded-full bg-cyan-500/10 flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-cyan-500" />
              </div>
              <span className="text-sm font-medium">Run Backtest</span>
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;
