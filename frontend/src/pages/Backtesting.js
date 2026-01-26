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
  Area,
  Legend
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
    initial_capital: 10000,
    position_size: 0.1,
    target_candle: 1
  });

  const [selectedModel, setSelectedModel] = useState(null);
  const [availableColumns, setAvailableColumns] = useState([]);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [modelsRes, filesRes, resultsRes] = await Promise.all([
        axios.get(`${API}/models/saved`),
        axios.get(`${API}/data/files`, { params: { lightweight: true } }),
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

  const handleModelSelect = (modelId) => {
    setConfig({ ...config, model_id: modelId });
    const model = savedModels.find(m => m.id === modelId);
    setSelectedModel(model);
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
    if (!config.model_id || !config.test_data_path) {
      toast.error("Please select a model and test data file");
      return;
    }

    if (!selectedModel) {
      toast.error("Model configuration not loaded");
      return;
    }

    setRunning(true);
    try {
      const response = await axios.post(`${API}/backtest/run`, config);
      setCurrentResult(response.data);
      setResults(prev => [response.data, ...prev]);
      toast.success("Backtest completed");
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to run backtest");
    } finally {
      setRunning(false);
    }
  };

  const MetricCard = ({ title, value, icon: Icon, positive, subtitle }) => (
    <div className="stat-card">
      <div className="flex items-start justify-between">
        <div>
          <p className="metric-label">{title}</p>
          <p className={`metric-value text-xl ${positive === true ? 'trading-positive' :
            positive === false ? 'trading-negative' : ''
            }`}>
            {value}
          </p>
          {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
        </div>
        <div className={`w-8 h-8 rounded-md flex items-center justify-center ${positive === true ? 'bg-green-500/10' :
          positive === false ? 'bg-red-500/10' : 'bg-primary/10'
          }`}>
          <Icon className={`w-4 h-4 ${positive === true ? 'text-green-500' :
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
                onValueChange={handleModelSelect}
              >
                <SelectTrigger data-testid="backtest-model-select">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {savedModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name} ({model.file_type === 'py' ? 'Python' : 'Joblib'})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {savedModels.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  No trained models. Train a model first.
                </p>
              )}
              {selectedModel && (
                <div className="mt-2 p-2 bg-accent rounded-md text-xs space-y-1">
                  <p className="font-semibold">Model Configuration:</p>
                  {selectedModel.target_column && (
                    <p><span className="text-muted-foreground">Target:</span> {selectedModel.target_column}</p>
                  )}
                  {selectedModel.feature_columns && selectedModel.feature_columns.length > 0 && (
                    <div>
                      <p className="text-muted-foreground">Features ({selectedModel.feature_columns.length}):</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {selectedModel.feature_columns.slice(0, 5).map((col, i) => (
                          <span key={i} className="px-1.5 py-0.5 bg-background rounded text-xs">
                            {col}
                          </span>
                        ))}
                        {selectedModel.feature_columns.length > 5 && (
                          <span className="px-1.5 py-0.5 bg-background rounded text-xs">
                            +{selectedModel.feature_columns.length - 5} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
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


            {/* Trading Parameters */}
            <div className="space-y-2">
              <Label>Target Candle</Label>
              <Select
                value={config.target_candle.toString()}
                onValueChange={(v) => setConfig({ ...config, target_candle: parseInt(v) })}
              >
                <SelectTrigger data-testid="target-candle-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((num) => (
                    <SelectItem key={num} value={num.toString()}>
                      {num === 1 ? '1st' : num === 2 ? '2nd' : num === 3 ? '3rd' : `${num}th`} Candle
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Target candle for hit detection (within OHLC range)
              </p>
            </div>

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
                  value={`${(currentResult?.total_return || 0) >= 0 ? '+' : ''}${(currentResult?.total_return || 0).toFixed(2)}%`}
                  icon={(currentResult?.total_return || 0) >= 0 ? TrendingUp : TrendingDown}
                  positive={(currentResult?.total_return || 0) >= 0}
                />
                <MetricCard
                  title="Sharpe Ratio"
                  value={(currentResult?.sharpe_ratio || 0).toFixed(2)}
                  icon={Target}
                  positive={(currentResult?.sharpe_ratio || 0) > 1}
                />
                <MetricCard
                  title="Max Drawdown"
                  value={`-${(currentResult?.max_drawdown || 0).toFixed(2)}%`}
                  icon={ArrowDownRight}
                  positive={false}
                />
                <MetricCard
                  title="Total Trades"
                  value={currentResult?.total_trades || 0}
                  icon={Activity}
                  subtitle={`Win: ${currentResult?.winning_trades || 0} | Loss: ${currentResult?.losing_trades || 0}`}
                />
              </div>

              {/* Price Action Chart */}
              <Card className="panel" data-testid="price-chart">
                <CardHeader className="panel-header py-3">
                  <CardTitle className="panel-title text-base font-medium flex items-center">
                    <LineChartIcon className="w-4 h-4 mr-2" />
                    Price Action & Signal Markers
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4 pt-2">
                  <div className="w-full h-[350px]">
                    {currentResult.price_data && currentResult.price_data.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={currentResult.price_data}>
                          <defs>
                            <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                          <XAxis
                            dataKey="time"
                            stroke="#525252"
                            tick={{ fill: '#a3a3a3', fontSize: 10 }}
                            minTickGap={50}
                            axisLine={false}
                            tickLine={false}
                          />
                          <YAxis
                            domain={['auto', 'auto']}
                            stroke="#525252"
                            tick={{ fill: '#a3a3a3', fontSize: 11 }}
                            axisLine={false}
                            tickLine={false}
                            tickFormatter={(val) => `$${val.toFixed(0)}`}
                          />
                          <Tooltip
                            contentStyle={{
                              background: '#0a0a0a',
                              border: '1px solid #262626',
                              borderRadius: '8px',
                              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
                              color: '#f5f5f5'
                            }}
                          />
                          <Legend verticalAlign="top" height={36} iconType="circle" />
                          <Area
                            type="linear"
                            dataKey="close"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorClose)"
                            name="Price"
                            isAnimationActive={false}
                          />
                          {/* BUY Markers */}
                          <Scatter
                            name="BUY Order"
                            data={currentResult.trades?.filter(t => t.type === 'BUY')}
                            dataKey="price"
                            fill="#22c55e"
                          >
                            {currentResult.trades?.filter(t => t.type === 'BUY').map((entry, index) => (
                              <Cell key={`buy-${index}`} fill="#22c55e" stroke="#fff" strokeWidth={0.5} r={1.5} />
                            ))}
                          </Scatter>
                          {/* SELL Markers */}
                          <Scatter
                            name="SELL Order"
                            data={currentResult.trades?.filter(t => t.type === 'SELL')}
                            dataKey="price"
                            fill="#ef4444"
                          >
                            {currentResult.trades?.filter(t => t.type === 'SELL').map((entry, index) => (
                              <Cell key={`sell-${index}`} fill="#ef4444" stroke="#fff" strokeWidth={0.5} r={1.5} />
                            ))}
                          </Scatter>
                        </ComposedChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex flex-col items-center justify-center h-full text-muted-foreground bg-accent/5 rounded-lg border border-dashed">
                        <LineChartIcon className="w-12 h-12 mb-2 opacity-50" />
                        <p>No price data available for visualization</p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Equity Curve Chart */}
              <Card className="panel" data-testid="equity-chart">
                <CardHeader className="panel-header py-3">
                  <CardTitle className="panel-title text-base font-medium flex items-center">
                    <Activity className="w-4 h-4 mr-2" />
                    Portfolio Equity Growth
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4 pt-2">
                  <div className="w-full h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={currentResult.price_data}>
                        <defs>
                          <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ec4899" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#ec4899" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                        <XAxis
                          dataKey="time"
                          hide={true}
                        />
                        <YAxis
                          domain={['auto', 'auto']}
                          stroke="#525252"
                          tick={{ fill: '#a3a3a3', fontSize: 10 }}
                          axisLine={false}
                          tickLine={false}
                          tickFormatter={(val) => `$${val.toLocaleString()}`}
                        />
                        <Tooltip
                          contentStyle={{
                            background: '#0a0a0a',
                            border: '1px solid #262626',
                            borderRadius: '8px',
                            color: '#f5f5f5'
                          }}
                          formatter={(value) => [`$${value.toLocaleString()}`, 'Equity']}
                        />
                        <Area
                          type="monotone"
                          dataKey={(obj) => {
                            const ret = currentResult?.total_return || 0;
                            return ret * 100 + 10000;
                          }}
                          stroke="#ec4899"
                          fillOpacity={1}
                          fill="url(#colorEquity)"
                          isAnimationActive={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Trade List */}

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
                              <span className={`badge ${trade.type === 'BUY' ? 'status-running' : 'status-failed'
                                }`}>
                                {trade.type}
                              </span>
                            </TableCell>
                            <TableCell className="font-mono text-xs">
                              {trade.time}
                            </TableCell>
                            <TableCell className="font-mono text-right">
                              ${(trade.price || 0).toFixed(2)}
                            </TableCell>
                            <TableCell className="font-mono text-right">
                              {(trade.shares || 0).toFixed(4)}
                            </TableCell>
                            <TableCell className="font-mono text-right">
                              ${(trade.value || 0).toFixed(2)}
                            </TableCell>
                            <TableCell className={`font-mono text-right ${(trade.pnl || 0) > 0 ? 'trading-positive' :
                              (trade.pnl || 0) < 0 ? 'trading-negative' : ''
                              }`}>
                              {trade.pnl !== undefined ?
                                `${(trade.pnl || 0) >= 0 ? '+' : ''}$${(trade.pnl || 0).toFixed(2)}` : '-'}
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
