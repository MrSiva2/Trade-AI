import { useState, useEffect } from "react";
import axios from "axios";
import { API } from "../App";
import { toast } from "sonner";
import {
  Brain,
  Plus,
  Download,
  Upload,
  Sparkles,
  Layers,
  Settings2,
  Check,
  Loader2,
  FolderInput,
  Save
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "../components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import { ScrollArea } from "../components/ui/scroll-area";

const ModelManagement = () => {
  const [prebuiltModels, setPrebuiltModels] = useState([]);
  const [customModels, setCustomModels] = useState([]);
  const [savedModels, setSavedModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importPath, setImportPath] = useState("");
  const [customConfig, setCustomConfig] = useState({
    name: "",
    layers: [
      { type: "dense", units: 64, activation: "relu" },
      { type: "dropout", rate: 0.2 },
      { type: "dense", units: 32, activation: "relu" },
      { type: "dense", units: 1, activation: "sigmoid" }
    ],
    optimizer: "adam",
    loss: "binary_crossentropy",
    learning_rate: 0.001
  });

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const [prebuilt, custom, saved] = await Promise.all([
        axios.get(`${API}/models/prebuilt`),
        axios.get(`${API}/models/custom`),
        axios.get(`${API}/models/saved`)
      ]);
      setPrebuiltModels(prebuilt.data.models || []);
      setCustomModels(custom.data.models || []);
      setSavedModels(saved.data.models || []);
    } catch (error) {
      toast.error("Failed to fetch models");
    } finally {
      setLoading(false);
    }
  };

  const createCustomModel = async () => {
    if (!customConfig.name.trim()) {
      toast.error("Please enter a model name");
      return;
    }

    try {
      await axios.post(`${API}/models/custom`, customConfig);
      toast.success("Custom model created successfully");
      setShowCreateDialog(false);
      fetchModels();
    } catch (error) {
      toast.error("Failed to create model");
    }
  };

  const importModels = async () => {
    if (!importPath.trim()) {
      toast.error("Please enter a folder path");
      return;
    }

    try {
      const response = await axios.post(`${API}/models/import`, { path: importPath });
      toast.success(`Imported ${response.data.imported?.length || 0} models`);
      setShowImportDialog(false);
      setImportPath("");
      fetchModels();
    } catch (error) {
      toast.error("Failed to import models");
    }
  };

  const addLayer = () => {
    setCustomConfig({
      ...customConfig,
      layers: [...customConfig.layers, { type: "dense", units: 32, activation: "relu" }]
    });
  };

  const updateLayer = (index, field, value) => {
    const newLayers = [...customConfig.layers];
    newLayers[index] = { ...newLayers[index], [field]: value };
    setCustomConfig({ ...customConfig, layers: newLayers });
  };

  const removeLayer = (index) => {
    setCustomConfig({
      ...customConfig,
      layers: customConfig.layers.filter((_, i) => i !== index)
    });
  };

  const ModelCard = ({ model, type }) => (
    <div
      className={`model-card ${selectedModel?.id === model.id ? 'model-card-selected' : ''}`}
      onClick={() => setSelectedModel(model)}
      data-testid={`model-card-${model.id}`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center">
          {type === "prebuilt" ? (
            <Sparkles className="w-5 h-5 text-primary" />
          ) : type === "custom" ? (
            <Layers className="w-5 h-5 text-purple-400" />
          ) : (
            <Save className="w-5 h-5 text-cyan-400" />
          )}
        </div>
        {selectedModel?.id === model.id && (
          <Check className="w-5 h-5 text-primary" />
        )}
      </div>
      <h3 className="font-semibold text-sm mb-1">{model.name}</h3>
      <p className="text-xs text-muted-foreground mb-2">
        {model.type?.replace(/_/g, ' ').toUpperCase()}
      </p>
      {model.description && (
        <p className="text-xs text-muted-foreground line-clamp-2">
          {model.description}
        </p>
      )}
      {model.parameters && (
        <div className="mt-3 pt-3 border-t border-border">
          <div className="flex flex-wrap gap-1">
            {Object.entries(model.parameters).slice(0, 3).map(([key, value]) => (
              <span
                key={key}
                className="px-1.5 py-0.5 text-xs font-mono bg-accent rounded"
              >
                {key}: {value}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]" data-testid="models-loading">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in" data-testid="model-management">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Model Management</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Select, create, or import trading AI models
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => setShowImportDialog(true)}
            data-testid="import-model-btn"
          >
            <FolderInput className="w-4 h-4 mr-2" />
            Import
          </Button>
          <Button onClick={() => setShowCreateDialog(true)} data-testid="create-model-btn">
            <Plus className="w-4 h-4 mr-2" />
            Create Custom
          </Button>
        </div>
      </div>

      {/* Selected Model Info */}
      {selectedModel && (
        <Card className="border-primary/50 bg-primary/5" data-testid="selected-model-info">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-md bg-primary/20 flex items-center justify-center">
                  <Brain className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold">{selectedModel.name}</h3>
                  <p className="text-sm text-muted-foreground">
                    Type: {selectedModel.type?.replace(/_/g, ' ')} | ID: {selectedModel.id}
                  </p>
                </div>
              </div>
              <Button onClick={() => window.location.href = `/training?model=${selectedModel.id}`}>
                Use for Training
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Tabs */}
      <Tabs defaultValue="prebuilt" className="space-y-4">
        <TabsList className="bg-card border border-border">
          <TabsTrigger value="prebuilt" data-testid="tab-prebuilt">
            <Sparkles className="w-4 h-4 mr-2" />
            Pre-built ({prebuiltModels.length})
          </TabsTrigger>
          <TabsTrigger value="custom" data-testid="tab-custom">
            <Layers className="w-4 h-4 mr-2" />
            Custom ({customModels.length})
          </TabsTrigger>
          <TabsTrigger value="saved" data-testid="tab-saved">
            <Save className="w-4 h-4 mr-2" />
            Saved ({savedModels.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="prebuilt">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {prebuiltModels.map((model) => (
              <ModelCard key={model.id} model={model} type="prebuilt" />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="custom">
          {customModels.length === 0 ? (
            <Card className="panel">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Layers className="w-12 h-12 mb-4 text-muted-foreground opacity-50" />
                <p className="text-muted-foreground mb-2">No custom models yet</p>
                <Button onClick={() => setShowCreateDialog(true)}>
                  <Plus className="w-4 h-4 mr-2" />
                  Create Custom Model
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {customModels.map((model) => (
                <ModelCard key={model.id} model={model} type="custom" />
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="saved">
          {savedModels.length === 0 ? (
            <Card className="panel">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Save className="w-12 h-12 mb-4 text-muted-foreground opacity-50" />
                <p className="text-muted-foreground mb-2">No saved models yet</p>
                <p className="text-sm text-muted-foreground">
                  Train a model to save it here
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {savedModels.map((model) => (
                <ModelCard key={model.id} model={model} type="saved" />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Create Custom Model Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="max-w-2xl" data-testid="create-model-dialog">
          <DialogHeader>
            <DialogTitle>Create Custom Model</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Model Name</Label>
              <Input
                placeholder="My Custom Model"
                value={customConfig.name}
                onChange={(e) => setCustomConfig({ ...customConfig, name: e.target.value })}
                data-testid="model-name-input"
              />
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>Optimizer</Label>
                <Select
                  value={customConfig.optimizer}
                  onValueChange={(v) => setCustomConfig({ ...customConfig, optimizer: v })}
                >
                  <SelectTrigger data-testid="optimizer-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="adam">Adam</SelectItem>
                    <SelectItem value="sgd">SGD</SelectItem>
                    <SelectItem value="rmsprop">RMSprop</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Loss Function</Label>
                <Select
                  value={customConfig.loss}
                  onValueChange={(v) => setCustomConfig({ ...customConfig, loss: v })}
                >
                  <SelectTrigger data-testid="loss-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="binary_crossentropy">Binary Crossentropy</SelectItem>
                    <SelectItem value="mse">MSE</SelectItem>
                    <SelectItem value="categorical_crossentropy">Categorical Crossentropy</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Learning Rate</Label>
                <Input
                  type="number"
                  step="0.0001"
                  value={customConfig.learning_rate}
                  onChange={(e) => setCustomConfig({ ...customConfig, learning_rate: parseFloat(e.target.value) })}
                  data-testid="learning-rate-input"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Layers</Label>
                <Button size="sm" variant="outline" onClick={addLayer} data-testid="add-layer-btn">
                  <Plus className="w-4 h-4 mr-1" />
                  Add Layer
                </Button>
              </div>
              <ScrollArea className="h-[200px] border border-border rounded-md p-3">
                <div className="space-y-3">
                  {customConfig.layers.map((layer, index) => (
                    <div key={index} className="flex items-center gap-2 p-2 bg-accent rounded-md">
                      <span className="text-xs text-muted-foreground w-6">#{index + 1}</span>
                      <Select
                        value={layer.type}
                        onValueChange={(v) => updateLayer(index, 'type', v)}
                      >
                        <SelectTrigger className="w-28 h-8">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="dense">Dense</SelectItem>
                          <SelectItem value="dropout">Dropout</SelectItem>
                          <SelectItem value="lstm">LSTM</SelectItem>
                          <SelectItem value="conv1d">Conv1D</SelectItem>
                        </SelectContent>
                      </Select>
                      {layer.type === 'dense' || layer.type === 'lstm' ? (
                        <>
                          <Input
                            type="number"
                            placeholder="Units"
                            value={layer.units || ''}
                            onChange={(e) => updateLayer(index, 'units', parseInt(e.target.value))}
                            className="w-20 h-8"
                          />
                          <Select
                            value={layer.activation || 'relu'}
                            onValueChange={(v) => updateLayer(index, 'activation', v)}
                          >
                            <SelectTrigger className="w-24 h-8">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="relu">ReLU</SelectItem>
                              <SelectItem value="sigmoid">Sigmoid</SelectItem>
                              <SelectItem value="tanh">Tanh</SelectItem>
                              <SelectItem value="softmax">Softmax</SelectItem>
                            </SelectContent>
                          </Select>
                        </>
                      ) : layer.type === 'dropout' ? (
                        <Input
                          type="number"
                          step="0.1"
                          placeholder="Rate"
                          value={layer.rate || ''}
                          onChange={(e) => updateLayer(index, 'rate', parseFloat(e.target.value))}
                          className="w-20 h-8"
                        />
                      ) : null}
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => removeLayer(index)}
                        className="ml-auto h-8 w-8 p-0"
                      >
                        ×
                      </Button>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button onClick={createCustomModel} data-testid="save-model-btn">
              Create Model
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Import Dialog */}
      <Dialog open={showImportDialog} onOpenChange={setShowImportDialog}>
        <DialogContent data-testid="import-model-dialog">
          <DialogHeader>
            <DialogTitle>Import Models</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Folder Path</Label>
              <Input
                placeholder="/path/to/models"
                value={importPath}
                onChange={(e) => setImportPath(e.target.value)}
                data-testid="import-path-input"
              />
              <p className="text-xs text-muted-foreground">
                Enter the path to a folder containing .joblib model files
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowImportDialog(false)}>
              Cancel
            </Button>
            <Button onClick={importModels} data-testid="confirm-import-btn">
              <Upload className="w-4 h-4 mr-2" />
              Import
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ModelManagement;
