import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { API } from "../App";
import { toast } from "sonner";
import {
  Play,
  LineChart as LineChartIcon,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Percent,
  ArrowUpRight,
  ArrowDownRight,
  Loader2,
  BarChart3,
  Target,
  Activity
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { ScrollArea } from "../components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Scatter,
  Cell,
  AreaChart,
  Area
} from "recharts";

const Backtesting = () => {
  const [savedModels, setSavedModels] = useState([]);
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [currentResult, setCurrentResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);

  const [config, setConfig] = useState({
    model_id: "",
    test_data_path: "",
    target_column: "",
    feature_columns: [],
    initial_capital: 10000,
    position_size: 0.1
  });

  const [availableColumns, setAvailableColumns] = useState([]);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [modelsRes, filesRes, resultsRes] = await Promise.all([
        axios.get(`${API}/models/saved`),
        axios.get(`${API}/data/files`),
        axios.get(`${API}/backtest/results`)
      ]);
      setSavedModels(modelsRes.data.models || []);
      setFiles(filesRes.data.files || []);
      setResults(resultsRes.data.results || []);
    } catch (error) {
      toast.error("Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = async (path) => {
    setConfig({ ...config, test_data_path: path });
    try {
      const response = await axios.get(`${API}/data/preview`, {
        params: { file_path: path, rows: 5 }
      });
      setAvailableColumns(response.data.columns || []);
    } catch (error) {
      console.error("Failed to get columns:", error);
    }
  };

  const runBacktest = async () => {
    if (!config.model_id || !config.test_data_path || !config.target_column || config.feature_columns.length === 0) {
      toast.error("Please fill in all required fields");
      return;
    }

    setRunning(true);
    try {
      const response = await axios.post(`${API}/backtest/run`, config);
      setCurrentResult(response.data);
      setResults(prev => [response.data, ...prev]);
      toast.success("Backtest completed");
    } catch (error) {
      toast.error("Failed to run backtest");
    } finally {
      setRunning(false);
    }
  };

  const MetricCard = ({ title, value, icon: Icon, positive, subtitle }) => (
    <div className="stat-card">
      <div className="flex items-start justify-between">
        <div>
          <p className="metric-label">{title}</p>
          <p className={`metric-value text-xl ${
            positive === true ? 'trading-positive' :
            positive === false ? 'trading-negative' : ''
          }`}>
            {value}
          </p>
          {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
        </div>
        <div className={`w-8 h-8 rounded-md flex items-center justify-center ${
          positive === true ? 'bg-green-500/10' :
          positive === false ? 'bg-red-500/10' : 'bg-primary/10'
        }`}>
          <Icon className={`w-4 h-4 ${
            positive === true ? 'text-green-500' :
            positive === false ? 'text-red-500' : 'text-primary'
          }`} />
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]" data-testid="backtest-loading">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in" data-testid="backtesting-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Backtesting</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Test your models against historical data
          </p>
        </div>
        <Button 
          onClick={runBacktest} 
          disabled={running}
          data-testid="run-backtest-btn"
        >
          {running ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Play className="w-4 h-4 mr-2" />
          )}
          Run Backtest
        </Button>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* Configuration Panel */}
        <Card className="col-span-12 lg:col-span-3 panel" data-testid="backtest-config-panel">
          <CardHeader className="panel-header">
            <CardTitle className="panel-title">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="p-4 space-y-4">
            {/* Model Selection */}
            <div className="space-y-2">
              <Label>Trained Model</Label>
              <Select
                value={config.model_id}
                onValueChange={(v) => setConfig({ ...config, model_id: v })}
              >
                <SelectTrigger data-testid="backtest-model-select">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {savedModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {savedModels.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  No trained models. Train a model first.
                </p>
              )}
            </div>

            {/* Test Data */}
            <div className="space-y-2">
              <Label>Test Data</Label>
              <Select
                value={config.test_data_path}
                onValueChange={handleFileSelect}
              >
                <SelectTrigger data-testid="backtest-data-select">
                  <SelectValue placeholder="Select CSV file" />
                </SelectTrigger>
                <SelectContent>
                  {files.map((file) => (
                    <SelectItem key={file.path} value={file.path}>
                      {file.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Target Column */}
            <div className="space-y-2">
              <Label>Target Column</Label>
              <Select
                value={config.target_column}
                onValueChange={(v) => setConfig({ ...config, target_column: v })}
              >
                <SelectTrigger data-testid="backtest-target-select">
                  <SelectValue placeholder="Select target" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Feature Columns */}
            <div className="space-y-2">
              <Label>Feature Columns</Label>
              <div className="flex flex-wrap gap-1 p-2 border border-border rounded-md min-h-[60px] max-h-[120px] overflow-y-auto">
                {availableColumns
                  .filter(col => col !== config.target_column)
                  .map((col) => (
                    <button
                      key={col}
                      onClick={() => {
                        const newFeatures = config.feature_columns.includes(col)
                          ? config.feature_columns.filter(c => c !== col)
                          : [...config.feature_columns, col];
                        setConfig({ ...config, feature_columns: newFeatures });
                      }}
                      className={`px-2 py-1 text-xs rounded transition-colors ${
                        config.feature_columns.includes(col)
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-accent hover:bg-accent/80'
                      }`}
                    >
                      {col}
                    </button>
                  ))}
              </div>
            </div>

            {/* Trading Parameters */}
            <div className="space-y-2">
              <Label>Initial Capital</Label>
              <Input
                type="number"
                value={config.initial_capital}
                onChange={(e) => setConfig({ ...config, initial_capital: parseFloat(e.target.value) })}
                data-testid="initial-capital-input"
              />
            </div>

            <div className="space-y-2">
              <Label>Position Size (%)</Label>
              <Input
                type="number"
                step="0.01"
                min="0.01"
                max="1"
                value={config.position_size}
                onChange={(e) => setConfig({ ...config, position_size: parseFloat(e.target.value) })}
                data-testid="position-size-input"
              />
            </div>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <div className="col-span-12 lg:col-span-9 space-y-4">
          {currentResult ? (
            <>
              {/* Metrics Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4" data-testid="backtest-metrics">
                <MetricCard
                  title="Total Return"
                  value={`${currentResult.total_return >= 0 ? '+' : ''}${currentResult.total_return.toFixed(2)}%`}
                  icon={currentResult.total_return >= 0 ? TrendingUp : TrendingDown}
                  positive={currentResult.total_return >= 0}
                />
                <MetricCard
                  title="Sharpe Ratio"
                  value={currentResult.sharpe_ratio.toFixed(2)}
                  icon={Target}
                  positive={currentResult.sharpe_ratio > 1}
                />
                <MetricCard
                  title="Max Drawdown"
                  value={`-${currentResult.max_drawdown.toFixed(2)}%`}
                  icon={ArrowDownRight}
                  positive={false}
                />
                <MetricCard
                  title="Total Trades"
                  value={currentResult.total_trades}
                  icon={Activity}
                  subtitle={`Win: ${currentResult.winning_trades} | Loss: ${currentResult.losing_trades}`}
                />
              </div>

              {/* TradingView-style Chart */}
              <Card className="panel" data-testid="backtest-chart">
                <CardHeader className="panel-header">
                  <CardTitle className="panel-title">
                    <LineChartIcon className="w-4 h-4 mr-2 inline" />
                    Price Chart with Trade Executions
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div ref={chartContainerRef} className="w-full" />
                </CardContent>
              </Card>

              {/* Trade List */}
              <Card className="panel" data-testid="trade-list">
                <CardHeader className="panel-header">
                  <CardTitle className="panel-title">Trade History</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <ScrollArea className="h-[200px]">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Type</TableHead>
                          <TableHead>Time</TableHead>
                          <TableHead className="text-right">Price</TableHead>
                          <TableHead className="text-right">Shares</TableHead>
                          <TableHead className="text-right">Value</TableHead>
                          <TableHead className="text-right">P&L</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {currentResult.trades?.map((trade, i) => (
                          <TableRow key={i}>
                            <TableCell>
                              <span className={`badge ${
                                trade.type === 'BUY' ? 'status-running' : 'status-failed'
                              }`}>
                                {trade.type}
                              </span>
                            </TableCell>
                            <TableCell className="font-mono text-xs">
                              {trade.time}
                            </TableCell>
                            <TableCell className="font-mono text-right">
                              ${trade.price.toFixed(2)}
                            </TableCell>
                            <TableCell className="font-mono text-right">
                              {trade.shares.toFixed(4)}
                            </TableCell>
                            <TableCell className="font-mono text-right">
                              ${trade.value.toFixed(2)}
                            </TableCell>
                            <TableCell className={`font-mono text-right ${
                              trade.pnl > 0 ? 'trading-positive' : 
                              trade.pnl < 0 ? 'trading-negative' : ''
                            }`}>
                              {trade.pnl !== undefined ? 
                                `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}` : '-'}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card className="panel h-[600px]">
              <CardContent className="flex flex-col items-center justify-center h-full text-muted-foreground">
                <BarChart3 className="w-16 h-16 mb-4 opacity-50" />
                <p className="text-lg">No backtest results yet</p>
                <p className="text-sm">Configure parameters and run a backtest</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default Backtesting;
