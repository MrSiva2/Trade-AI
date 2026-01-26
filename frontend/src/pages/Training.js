import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import { API } from "../App";
import { toast } from "sonner";
import {
  Play,
  Square,
  RefreshCw,
  Brain,
  FileSpreadsheet,
  Settings2,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
  Terminal,
  BarChart3
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Progress } from "../components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { ScrollArea } from "../components/ui/scroll-area";
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
  Legend
} from "recharts";

const Training = () => {
  const [models, setModels] = useState([]);
  const [files, setFiles] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [logs, setLogs] = useState([]);
  const [outputs, setOutputs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const logsEndRef = useRef(null);
  const outputsEndRef = useRef(null);

  const [config, setConfig] = useState({
    model_id: "",
    model_name: "",
    train_data_path: "",
    test_data_path: "",
    target_column: "",
    feature_columns: [],
    epochs: 100,
    batch_size: 32,
    validation_split: 0.2
  });

  const [availableColumns, setAvailableColumns] = useState([]);

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (activeSession && (activeSession.status === "pending" || activeSession.status === "running")) {
      const interval = setInterval(() => {
        fetchSessionStatus(activeSession.id);
        fetchLogs(activeSession.id);
        fetchOutputs(activeSession.id);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [activeSession]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    outputsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [outputs]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [modelsRes, filesRes, sessionsRes] = await Promise.all([
        axios.get(`${API}/models/prebuilt`),
        axios.get(`${API}/data/files`, { params: { lightweight: true } }),
        axios.get(`${API}/training/sessions`)
      ]);
      setModels(modelsRes.data.models || []);
      setFiles(filesRes.data.files || []);
      setSessions(sessionsRes.data.sessions || []);
    } catch (error) {
      toast.error("Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  const fetchSessionStatus = async (sessionId) => {
    try {
      const response = await axios.get(`${API}/training/status/${sessionId}`);
      setActiveSession(response.data);

      if (response.data.status === "completed" || response.data.status === "failed") {
        setTraining(false);
        toast.success(`Training ${response.data.status}`);
      }
    } catch (error) {
      console.error("Failed to fetch session status:", error);
    }
  };

  const fetchLogs = async (sessionId) => {
    try {
      const response = await axios.get(`${API}/training/logs/${sessionId}`);
      if (response.data.logs?.length > 0) {
        setLogs(prev => [...prev, ...response.data.logs]);
      }
    } catch (error) {
      console.error("Failed to fetch logs:", error);
    }
  };

  const fetchOutputs = async (sessionId) => {
    try {
      const response = await axios.get(`${API}/training/output/${sessionId}`);
      if (response.data.outputs?.length > 0) {
        setOutputs(prev => [...prev, ...response.data.outputs]);
      }
    } catch (error) {
      console.error("Failed to fetch outputs:", error);
    }
  };

  const handleFileSelect = async (path) => {
    setConfig({ ...config, train_data_path: path });
    try {
      const response = await axios.get(`${API}/data/preview`, {
        params: { file_path: path, rows: 5 }
      });
      setAvailableColumns(response.data.columns || []);
    } catch (error) {
      console.error("Failed to get columns:", error);
    }
  };

  const startTraining = async () => {
    if (!config.model_id || !config.train_data_path || !config.target_column || config.feature_columns.length === 0) {
      toast.error("Please fill in all required fields");
      return;
    }

    setTraining(true);
    setLogs([]);
    setOutputs([]);

    try {
      const response = await axios.post(`${API}/training/start`, config);
      setActiveSession({ id: response.data.session_id, status: "pending" });
      toast.success("Training started");
    } catch (error) {
      toast.error("Failed to start training");
      setTraining(false);
    }
  };

  const stopTraining = async () => {
    if (!activeSession) return;

    try {
      await axios.post(`${API}/training/stop/${activeSession.id}`);
      toast.info("Training stopped");
      setTraining(false);
    } catch (error) {
      toast.error("Failed to stop training");
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case "running":
        return <span className="badge status-running">Running</span>;
      case "completed":
        return <span className="badge status-completed">Completed</span>;
      case "failed":
        return <span className="badge status-failed">Failed</span>;
      case "stopped":
        return <span className="badge status-failed">Stopped</span>;
      default:
        return <span className="badge status-pending">Pending</span>;
    }
  };

  const chartData = activeSession?.metrics?.loss?.map((loss, i) => ({
    epoch: i + 1,
    loss: loss,
    accuracy: activeSession.metrics.accuracy?.[i] || 0,
    val_loss: activeSession.metrics.val_loss?.[i] || 0,
    val_accuracy: activeSession.metrics.val_accuracy?.[i] || 0
  })) || [];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]" data-testid="training-loading">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in" data-testid="training-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Training</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Train your trading AI models
          </p>
        </div>
        <div className="flex items-center gap-2">
          {training ? (
            <Button variant="destructive" onClick={stopTraining} data-testid="stop-training-btn">
              <Square className="w-4 h-4 mr-2" />
              Stop Training
            </Button>
          ) : (
            <Button onClick={startTraining} data-testid="start-training-btn">
              <Play className="w-4 h-4 mr-2" />
              Start Training
            </Button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* Configuration Panel */}
        <Card className="col-span-12 lg:col-span-4 panel" data-testid="config-panel">
          <CardHeader className="panel-header">
            <CardTitle className="panel-title">
              <Settings2 className="w-4 h-4 mr-2 inline" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 space-y-4">
            {/* Model Selection */}
            <div className="space-y-2">
              <Label>Model</Label>
              <Select
                value={config.model_id}
                onValueChange={(v) => setConfig({ ...config, model_id: v })}
              >
                <SelectTrigger data-testid="model-select">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Model Name */}
            <div className="space-y-2">
              <Label>Save Model As</Label>
              <Input
                placeholder="Enter model name..."
                value={config.model_name}
                onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                data-testid="model-name-input"
              />
              <p className="text-[10px] text-muted-foreground">
                Optional: If empty, a default name will be generated.
              </p>
            </div>

            {/* Training Data */}
            <div className="space-y-2">
              <Label>Training Data</Label>
              <Select
                value={config.train_data_path}
                onValueChange={handleFileSelect}
              >
                <SelectTrigger data-testid="train-data-select">
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
                <SelectTrigger data-testid="target-column-select">
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
              <div className="flex flex-wrap gap-1 p-2 border border-border rounded-md min-h-[60px]">
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
              <p className="text-xs text-muted-foreground">
                Selected: {config.feature_columns.length} columns
              </p>
            </div>

            {/* Hyperparameters */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Epochs</Label>
                <Input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                  data-testid="epochs-input"
                />
              </div>
              <div className="space-y-2">
                <Label>Batch Size</Label>
                <Input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                  data-testid="batch-size-input"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Validation Split</Label>
              <Input
                type="number"
                step="0.1"
                min="0"
                max="0.5"
                value={config.validation_split}
                onChange={(e) => setConfig({ ...config, validation_split: parseFloat(e.target.value) })}
                data-testid="validation-split-input"
              />
            </div>
          </CardContent>
        </Card>

        {/* Training Progress & Charts */}
        <div className="col-span-12 lg:col-span-8 space-y-4">
          {/* Progress Card */}
          {activeSession && (
            <Card className="panel" data-testid="progress-panel">
              <CardHeader className="panel-header">
                <CardTitle className="panel-title">Training Progress</CardTitle>
                {getStatusBadge(activeSession.status)}
              </CardHeader>
              <CardContent className="p-4">
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">
                      Epoch {activeSession.current_epoch} / {activeSession.total_epochs}
                    </span>
                    <span className="font-mono text-sm">
                      {((activeSession.current_epoch / activeSession.total_epochs) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    value={(activeSession.current_epoch / activeSession.total_epochs) * 100}
                    className="h-2"
                  />

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-4 gap-4 pt-4">
                    <div>
                      <p className="metric-label">Loss</p>
                      <p className="font-mono text-lg">
                        {activeSession.metrics?.loss?.slice(-1)[0]?.toFixed(4) || '-'}
                      </p>
                    </div>
                    <div>
                      <p className="metric-label">Accuracy</p>
                      <p className="font-mono text-lg trading-positive">
                        {activeSession.metrics?.accuracy?.slice(-1)[0]?.toFixed(4) || '-'}
                      </p>
                    </div>
                    <div>
                      <p className="metric-label">Val Loss</p>
                      <p className="font-mono text-lg">
                        {activeSession.metrics?.val_loss?.slice(-1)[0]?.toFixed(4) || '-'}
                      </p>
                    </div>
                    <div>
                      <p className="metric-label">Val Accuracy</p>
                      <p className="font-mono text-lg trading-positive">
                        {activeSession.metrics?.val_accuracy?.slice(-1)[0]?.toFixed(4) || '-'}
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Charts */}
          <Card className="panel" data-testid="charts-panel">
            <CardHeader className="panel-header">
              <CardTitle className="panel-title">Training Metrics</CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <div className="h-[300px]">
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
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
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={false}
                        name="Training Loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#22c55e"
                        strokeWidth={2}
                        dot={false}
                        name="Training Accuracy"
                      />
                      <Line
                        type="monotone"
                        dataKey="val_loss"
                        stroke="#f97316"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                        name="Validation Loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="val_accuracy"
                        stroke="#06b6d4"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                        name="Validation Accuracy"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                    <BarChart3 className="w-12 h-12 mb-2 opacity-50" />
                    <p>No training data yet</p>
                    <p className="text-sm">Start training to see metrics</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Logs & Output Tabs */}
          <Card className="panel" data-testid="logs-output-panel">
            <Tabs defaultValue="logs">
              <CardHeader className="panel-header">
                <TabsList className="bg-accent">
                  <TabsTrigger value="logs" data-testid="logs-tab">
                    <Terminal className="w-4 h-4 mr-2" />
                    Logs
                  </TabsTrigger>
                  <TabsTrigger value="output" data-testid="output-tab">
                    <Zap className="w-4 h-4 mr-2" />
                    Model Output
                  </TabsTrigger>
                </TabsList>
              </CardHeader>
              <CardContent className="p-0">
                <TabsContent value="logs" className="m-0">
                  <ScrollArea className="h-[200px]">
                    <div className="p-2 font-mono text-xs space-y-1">
                      {logs.length === 0 ? (
                        <p className="text-muted-foreground p-2">No logs yet...</p>
                      ) : (
                        logs.map((log, i) => (
                          <div
                            key={i}
                            className={`log-entry ${log.level === 'ERROR' ? 'log-error' :
                              log.level === 'WARNING' ? 'log-warning' :
                                log.level === 'SUCCESS' ? 'log-success' : 'log-info'
                              }`}
                          >
                            <span className="text-muted-foreground mr-2">
                              [{new Date(log.timestamp).toLocaleTimeString()}]
                            </span>
                            <span className="mr-2">[{log.level}]</span>
                            {log.message}
                          </div>
                        ))
                      )}
                      <div ref={logsEndRef} />
                    </div>
                  </ScrollArea>
                </TabsContent>
                <TabsContent value="output" className="m-0">
                  <ScrollArea className="h-[200px]">
                    <div className="p-2 font-mono text-xs space-y-2">
                      {outputs.length === 0 ? (
                        <p className="text-muted-foreground p-2">No model output yet...</p>
                      ) : (
                        outputs.map((output, i) => (
                          <div key={i} className="output-entry">
                            <div className="text-primary mb-1">Epoch {output.epoch}</div>
                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <span className="text-muted-foreground">Predictions: </span>
                                <span className="text-cyan-400">
                                  [{output.predictions?.slice(0, 5).join(', ')}...]
                                </span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Actual: </span>
                                <span className="text-green-400">
                                  [{output.actual?.slice(0, 5).join(', ')}...]
                                </span>
                              </div>
                            </div>
                            {output.probabilities?.length > 0 && (
                              <div className="mt-1">
                                <span className="text-muted-foreground">Probabilities: </span>
                                <span className="text-purple-400">
                                  [{output.probabilities?.slice(0, 5).map(p => p.toFixed(3)).join(', ')}...]
                                </span>
                              </div>
                            )}
                          </div>
                        ))
                      )}
                      <div ref={outputsEndRef} />
                    </div>
                  </ScrollArea>
                </TabsContent>
              </CardContent>
            </Tabs>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Training;
