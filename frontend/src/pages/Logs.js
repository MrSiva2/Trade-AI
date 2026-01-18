import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { API } from "../App";
import {
  Terminal,
  RefreshCw,
  Download,
  Filter,
  Search,
  Clock,
  Loader2,
  FileText,
  Activity
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { ScrollArea } from "../components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";

const Logs = () => {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [logs, setLogs] = useState([]);
  const [outputs, setOutputs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const logsEndRef = useRef(null);
  const outputsEndRef = useRef(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    if (selectedSession) {
      fetchLogs(selectedSession.id);
      fetchOutputs(selectedSession.id);
      
      if (selectedSession.status === "running") {
        const interval = setInterval(() => {
          fetchLogs(selectedSession.id);
          fetchOutputs(selectedSession.id);
        }, 1000);
        return () => clearInterval(interval);
      }
    }
  }, [selectedSession]);

  const fetchSessions = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/training/sessions`);
      setSessions(response.data.sessions || []);
      if (response.data.sessions?.length > 0) {
        setSelectedSession(response.data.sessions[0]);
      }
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    } finally {
      setLoading(false);
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

  const handleSessionChange = (sessionId) => {
    const session = sessions.find(s => s.id === sessionId);
    setSelectedSession(session);
    setLogs([]);
    setOutputs([]);
  };

  const filteredLogs = logs.filter(log => {
    if (filter !== "all" && log.level?.toLowerCase() !== filter) return false;
    if (searchTerm && !log.message?.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  const downloadLogs = () => {
    const content = logs.map(log => 
      `[${log.timestamp}] [${log.level}] ${log.message}`
    ).join('\n');
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_logs_${selectedSession?.id || 'unknown'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
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

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]" data-testid="logs-loading">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in" data-testid="logs-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Logs & Output</h1>
          <p className="text-muted-foreground text-sm mt-1">
            View training logs and model outputs
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={fetchSessions} data-testid="refresh-logs-btn">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button variant="outline" onClick={downloadLogs} data-testid="download-logs-btn">
            <Download className="w-4 h-4 mr-2" />
            Export Logs
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* Sessions List */}
        <Card className="col-span-12 lg:col-span-3 panel" data-testid="sessions-panel">
          <CardHeader className="panel-header">
            <CardTitle className="panel-title">Training Sessions</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[500px]">
              {sessions.length === 0 ? (
                <div className="p-4 text-center text-muted-foreground">
                  <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No training sessions yet</p>
                </div>
              ) : (
                sessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => handleSessionChange(session.id)}
                    className={`w-full p-3 text-left border-b border-border hover:bg-accent transition-colors ${
                      selectedSession?.id === session.id ? 'bg-accent' : ''
                    }`}
                    data-testid={`session-${session.id}`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-xs truncate max-w-[120px]">
                        {session.id.slice(0, 8)}...
                      </span>
                      {getStatusBadge(session.status)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Epoch {session.current_epoch}/{session.total_epochs}
                    </div>
                    {session.started_at && (
                      <div className="flex items-center gap-1 text-xs text-muted-foreground mt-1">
                        <Clock className="w-3 h-3" />
                        {new Date(session.started_at).toLocaleString()}
                      </div>
                    )}
                  </button>
                ))
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Logs & Output Panel */}
        <Card className="col-span-12 lg:col-span-9 panel" data-testid="logs-content-panel">
          <Tabs defaultValue="logs" className="h-full">
            <CardHeader className="panel-header">
              <TabsList className="bg-accent">
                <TabsTrigger value="logs" data-testid="logs-tab">
                  <Terminal className="w-4 h-4 mr-2" />
                  Training Logs
                </TabsTrigger>
                <TabsTrigger value="output" data-testid="output-tab">
                  <FileText className="w-4 h-4 mr-2" />
                  Model Output
                </TabsTrigger>
              </TabsList>
            </CardHeader>

            <TabsContent value="logs" className="m-0 h-full">
              {/* Filter Bar */}
              <div className="flex gap-2 p-3 border-b border-border">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    placeholder="Search logs..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-9"
                    data-testid="search-logs-input"
                  />
                </div>
                <Select value={filter} onValueChange={setFilter}>
                  <SelectTrigger className="w-32" data-testid="filter-select">
                    <Filter className="w-4 h-4 mr-2" />
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="info">Info</SelectItem>
                    <SelectItem value="success">Success</SelectItem>
                    <SelectItem value="warning">Warning</SelectItem>
                    <SelectItem value="error">Error</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <ScrollArea className="h-[400px]">
                <div className="p-2 font-mono text-xs space-y-0.5">
                  {filteredLogs.length === 0 ? (
                    <p className="text-muted-foreground p-4 text-center">
                      {selectedSession ? "No logs matching filter..." : "Select a session to view logs"}
                    </p>
                  ) : (
                    filteredLogs.map((log, i) => (
                      <div 
                        key={i} 
                        className={`log-entry ${
                          log.level === 'ERROR' ? 'log-error' :
                          log.level === 'WARNING' ? 'log-warning' :
                          log.level === 'SUCCESS' ? 'log-success' : 'log-info'
                        }`}
                      >
                        <span className="text-muted-foreground mr-2">
                          [{new Date(log.timestamp).toLocaleTimeString()}]
                        </span>
                        <span className={`mr-2 px-1 rounded text-xs ${
                          log.level === 'ERROR' ? 'bg-red-500/20' :
                          log.level === 'WARNING' ? 'bg-yellow-500/20' :
                          log.level === 'SUCCESS' ? 'bg-green-500/20' : 'bg-blue-500/20'
                        }`}>
                          {log.level}
                        </span>
                        {log.message}
                      </div>
                    ))
                  )}
                  <div ref={logsEndRef} />
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="output" className="m-0 h-full">
              <ScrollArea className="h-[450px]">
                <div className="p-4 font-mono text-xs space-y-4">
                  {outputs.length === 0 ? (
                    <p className="text-muted-foreground text-center">
                      {selectedSession ? "No model output yet..." : "Select a session to view output"}
                    </p>
                  ) : (
                    outputs.map((output, i) => (
                      <div key={i} className="p-3 bg-accent/50 rounded-md border border-border">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-primary font-semibold">
                            Epoch {output.epoch}
                          </span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div>
                            <span className="text-muted-foreground block mb-1">Predictions:</span>
                            <div className="flex flex-wrap gap-1">
                              {output.predictions?.slice(0, 10).map((pred, j) => (
                                <span 
                                  key={j}
                                  className={`px-1.5 py-0.5 rounded ${
                                    pred === 1 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                                  }`}
                                >
                                  {pred}
                                </span>
                              ))}
                              {output.predictions?.length > 10 && (
                                <span className="text-muted-foreground">+{output.predictions.length - 10} more</span>
                              )}
                            </div>
                          </div>
                          <div>
                            <span className="text-muted-foreground block mb-1">Actual:</span>
                            <div className="flex flex-wrap gap-1">
                              {output.actual?.slice(0, 10).map((val, j) => (
                                <span 
                                  key={j}
                                  className={`px-1.5 py-0.5 rounded ${
                                    val === 1 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                                  }`}
                                >
                                  {val}
                                </span>
                              ))}
                              {output.actual?.length > 10 && (
                                <span className="text-muted-foreground">+{output.actual.length - 10} more</span>
                              )}
                            </div>
                          </div>
                        </div>
                        {output.probabilities?.length > 0 && (
                          <div className="mt-2">
                            <span className="text-muted-foreground block mb-1">Probabilities:</span>
                            <div className="flex flex-wrap gap-1">
                              {output.probabilities?.slice(0, 10).map((prob, j) => (
                                <span 
                                  key={j}
                                  className="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400"
                                >
                                  {prob.toFixed(3)}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                  <div ref={outputsEndRef} />
                </div>
              </ScrollArea>
            </TabsContent>
          </Tabs>
        </Card>
      </div>
    </div>
  );
};

export default Logs;
