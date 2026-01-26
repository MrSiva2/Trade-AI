import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { API } from "../App";
import { toast } from "sonner";
import { createChart } from "lightweight-charts";
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
    position_size: 0.1,
    target_candle: 1
  });

  const [availableColumns, setAvailableColumns] = useState([]);
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);

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
          <p className={`metric-value text-xl font-bold ${positive === true ? 'text-emerald-500' :
            positive === false ? 'text-rose-500' : ''
            }`}>
            {value}
          </p>
          {subtitle && <p className="text-[10px] text-muted-foreground mt-1 uppercase tracking-wider">{subtitle}</p>}
        </div>
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${positive === true ? 'bg-emerald-500/10' :
          positive === false ? 'bg-rose-500/10' : 'bg-primary/10'
          }`}>
          <Icon className={`w-4 h-4 ${positive === true ? 'text-emerald-500' :
            positive === false ? 'text-rose-500' : 'text-primary'
            }`} />
        </div>
      </div>
    </div>
  );

  // Initialize and update chart
  useEffect(() => {
    if (!chartContainerRef.current || !currentResult) return;

    // Create chart if it doesn't exist
    if (!chartRef.current) {
      chartRef.current = createChart(chartContainerRef.current, {
        layout: {
          background: { color: 'transparent' },
          textColor: '#d1d4dc',
        },
        grid: {
          vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
          horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
        },
        rightPriceScale: {
          borderVisible: false,
        },
        timeScale: {
          borderVisible: false,
          timeVisible: true,
          secondsVisible: false,
        },
        handleScroll: true,
        handleScale: true,
      });

      candlestickSeriesRef.current = chartRef.current.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });

      const handleResize = () => {
        if (chartRef.current && chartContainerRef.current) {
          chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
        }
      };

      window.addEventListener('resize', handleResize);
      return () => {
        window.removeEventListener('resize', handleResize);
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
        }
      };
    }

    // Update chart data
    if (currentResult.price_data && currentResult.price_data.length > 0) {
      const formattedData = currentResult.price_data.map(d => ({
        time: (typeof d.time === 'string' && d.time.includes('-')) ? d.time :
          new Date(d.time).getTime() / 1000,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));

      // Sort data for lightweight-charts
      formattedData.sort((a, b) => {
        const timeA = typeof a.time === 'number' ? a.time : new Date(a.time).getTime();
        const timeB = typeof b.time === 'number' ? b.time : new Date(b.time).getTime();
        return timeA - timeB;
      });

      candlestickSeriesRef.current.setData(formattedData);

      // Add markers for trades
      const markers = currentResult.trades.map(trade => {
        return {
          time: (typeof trade.time === 'string' && trade.time.includes('-')) ? trade.time :
            new Date(trade.time).getTime() / 1000,
          position: trade.type === 'BUY' ? 'belowBar' : 'aboveBar',
          color: trade.type === 'BUY' ? '#26a69a' : '#ef5350',
          shape: trade.type === 'BUY' ? 'arrowUp' : 'arrowDown',
          text: trade.type === 'BUY' ? 'BUY' : `SELL ${trade.pnl >= 0 ? '✓' : '✗'}`,
        };
      });

      // Filter markers to ensure they exist in price data
      const validTimes = new Set(formattedData.map(d => d.time));
      const filteredMarkers = markers.filter(m => validTimes.has(m.time));

      candlestickSeriesRef.current.setMarkers(filteredMarkers);
      chartRef.current.timeScale().fitContent();
    }
  }, [currentResult]);

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
                      className={`px-2 py-1 text-xs rounded transition-colors ${config.feature_columns.includes(col)
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

        {/* Results / Dashboard Panel */}
        <div className="col-span-12 lg:col-span-9 space-y-4">
          {currentResult ? (
            <div className="grid grid-cols-12 gap-4">
              {/* Left Column: Main Chart */}
              <div className="col-span-12 xl:col-span-8 space-y-4">
                <Card className="panel border-primary/20" data-testid="backtest-chart">
                  <CardHeader className="panel-header flex flex-row items-center justify-between">
                    <CardTitle className="panel-title flex items-center">
                      <LineChartIcon className="w-4 h-4 mr-2 text-primary" />
                      Price Action & Predictions
                    </CardTitle>
                    <div className="flex items-center gap-4 text-xs font-mono">
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 bg-[#26a69a]" /> <span>Bullish</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 bg-[#ef5350]" /> <span>Bearish</span>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-0 bg-[#0b0e14]">
                    <div ref={chartContainerRef} className="w-full h-[500px]" />
                  </CardContent>
                </Card>

                {/* Trade List Table */}
                <Card className="panel" data-testid="trade-list">
                  <CardHeader className="panel-header">
                    <CardTitle className="panel-title">Trade History</CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <ScrollArea className="h-[250px]">
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
                                <span className={`badge ${trade.type === 'BUY' ? 'status-running' : 'status-failed'}`}>
                                  {trade.type}
                                </span>
                              </TableCell>
                              <TableCell className="font-mono text-xs">{trade.time}</TableCell>
                              <TableCell className="font-mono text-right">${trade.price.toFixed(2)}</TableCell>
                              <TableCell className="font-mono text-right">{trade.shares.toFixed(4)}</TableCell>
                              <TableCell className="font-mono text-right">${trade.value.toFixed(2)}</TableCell>
                              <TableCell className={`font-mono text-right ${trade.pnl > 0 ? 'trading-positive' : trade.pnl < 0 ? 'trading-negative' : ''}`}>
                                {trade.pnl !== undefined ? `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}` : '-'}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </div>

              {/* Right Column: Performance Metrics & Equity */}
              <div className="col-span-12 xl:col-span-4 space-y-4">
                <Card className="panel h-full">
                  <CardHeader className="panel-header">
                    <CardTitle className="panel-title flex items-center">
                      <Target className="w-4 h-4 mr-2 text-primary" />
                      Performance Analytics
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-4 space-y-4">
                    <div className="grid grid-cols-1 gap-3">
                      <MetricCard
                        title="Total Return"
                        value={`${currentResult.total_return >= 0 ? '+' : ''}${currentResult.total_return.toFixed(2)}%`}
                        icon={currentResult.total_return >= 0 ? TrendingUp : TrendingDown}
                        positive={currentResult.total_return >= 0}
                      />
                      <MetricCard
                        title="Win Rate"
                        value={`${((currentResult.winning_trades / (currentResult.total_trades || 1)) * 100).toFixed(1)}%`}
                        icon={Percent}
                        positive={currentResult.winning_trades / (currentResult.total_trades || 1) > 0.5}
                        subtitle={`${currentResult.winning_trades} Wins / ${currentResult.losing_trades} Losses`}
                      />
                      <MetricCard
                        title="Sharpe Ratio"
                        value={currentResult.sharpe_ratio.toFixed(2)}
                        icon={Activity}
                        positive={currentResult.sharpe_ratio > 1}
                      />
                      <MetricCard
                        title="Max Drawdown"
                        value={`-${currentResult.max_drawdown.toFixed(2)}%`}
                        icon={ArrowDownRight}
                        positive={false}
                      />
                    </div>

                    <div className="mt-6 pt-6 border-t border-border">
                      <h4 className="text-sm font-medium mb-4 flex items-center">
                        <BarChart3 className="w-4 h-4 mr-2" />
                        Equity Growth
                      </h4>
                      <div className="h-[200px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={currentResult.trades.filter(t => t.type === 'SELL').map((t, i) => ({ i, val: t.value }))}>
                            <defs>
                              <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="var(--primary)" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <Area type="monotone" dataKey="val" stroke="var(--primary)" fillOpacity={1} fill="url(#colorVal)" />
                            <Tooltip
                              contentStyle={{ background: '#0b0e14', border: '1px solid #2d3139' }}
                              labelFormatter={() => 'Trade Step'}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
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
