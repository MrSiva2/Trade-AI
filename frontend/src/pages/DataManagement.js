import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { API } from "../App";
import { toast } from "sonner";
import {
  FolderOpen,
  FileSpreadsheet,
  Upload,
  RefreshCw,
  Eye,
  ChevronRight,
  Loader2,
  HardDrive,
  Plus
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import { ScrollArea } from "../components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "../components/ui/dialog";

const DataManagement = () => {
  const [folders, setFolders] = useState([]);
  const [files, setFiles] = useState([]);
  const [selectedFolder, setSelectedFolder] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [newFolderPath, setNewFolderPath] = useState("");
  const [showPreview, setShowPreview] = useState(false);

  useEffect(() => {
    fetchFolders();
  }, []);

  const fetchFolders = async () => {
    try {
      const response = await axios.get(`${API}/data/folders`);
      setFolders(response.data.folders || []);
      if (response.data.folders?.length > 0) {
        selectFolder(response.data.folders[0]);
      }
    } catch (error) {
      toast.error("Failed to fetch folders");
    } finally {
      setLoading(false);
    }
  };

  const selectFolder = async (folder) => {
    setSelectedFolder(folder);
    setLoading(true);
    try {
      const response = await axios.get(`${API}/data/files`, {
        params: { folder: folder.path }
      });
      setFiles(response.data.files || []);
    } catch (error) {
      toast.error("Failed to fetch files");
    } finally {
      setLoading(false);
    }
  };

  const previewFile = async (file) => {
    setSelectedFile(file);
    setLoading(true);
    try {
      const response = await axios.get(`${API}/data/preview`, {
        params: { file_path: file.path, rows: 100 }
      });
      setPreviewData(response.data);
      setShowPreview(true);
    } catch (error) {
      toast.error("Failed to preview file");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    let csvCount = 0;

    for (let i = 0; i < files.length; i++) {
      if (files[i].name.toLowerCase().endsWith('.csv')) {
        formData.append('files', files[i]);
        csvCount++;
      }
    }

    if (csvCount === 0) {
      toast.error("No CSV files selected");
      return;
    }

    setUploading(true);
    try {
      const response = await axios.post(`${API}/data/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const { uploaded, failed, count } = response.data;

      if (count > 0) {
        toast.success(`Successfully uploaded ${count} file${count !== 1 ? 's' : ''}`);
      }

      if (failed && failed.length > 0) {
        toast.warning(`Failed to upload ${failed.length} file${failed.length !== 1 ? 's' : ''}`);
        console.warn("Failed uploads:", failed);
      }

      if (selectedFolder) {
        selectFolder(selectedFolder);
      }
    } catch (error) {
      console.error("Upload error:", error);
      toast.error("Failed to upload files");
    } finally {
      setUploading(false);
      // Reset input
      event.target.value = '';
    }
  };

  const addFolder = async () => {
    if (!newFolderPath.trim()) return;

    try {
      await axios.post(`${API}/data/set-folder`, { path: newFolderPath });
      toast.success("Folder added successfully");
      setNewFolderPath("");
      fetchFolders();
    } catch (error) {
      toast.error("Failed to add folder");
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="space-y-6 animate-fade-in" data-testid="data-management">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Data Management</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Manage your training and testing datasets
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => selectedFolder && selectFolder(selectedFolder)}
            data-testid="refresh-files-btn"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>

          {/* Folder Upload */}
          <label>
            <input
              type="file"
              directory=""
              webkitdirectory=""
              onChange={handleUpload}
              className="hidden"
              data-testid="folder-upload-input"
            />
            <Button variant="outline" asChild disabled={uploading}>
              <span className="cursor-pointer">
                <FolderOpen className="w-4 h-4 mr-2" />
                Upload Folder
              </span>
            </Button>
          </label>

          {/* Files Upload */}
          <label>
            <input
              type="file"
              accept=".csv"
              multiple
              onChange={handleUpload}
              className="hidden"
              data-testid="file-upload-input"
            />
            <Button asChild disabled={uploading}>
              <span className="cursor-pointer" data-testid="upload-btn">
                {uploading ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Upload className="w-4 h-4 mr-2" />
                )}
                Upload CSVs
              </span>
            </Button>
          </label>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* Folder Panel */}
        <Card className="col-span-12 md:col-span-3 panel" data-testid="folder-panel">
          <CardHeader className="panel-header">
            <CardTitle className="panel-title">Folders</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[300px]">
              {folders.map((folder, index) => (
                <button
                  key={index}
                  onClick={() => selectFolder(folder)}
                  className={`file-item w-full ${selectedFolder?.path === folder.path ? 'file-item-selected' : ''
                    }`}
                  data-testid={`folder-item-${index}`}
                >
                  <FolderOpen className="w-5 h-5 text-primary" />
                  <span className="truncate text-sm">{folder.name}</span>
                </button>
              ))}
            </ScrollArea>
            <div className="p-3 border-t border-border">
              <div className="flex gap-2">
                <Input
                  placeholder="Add folder path..."
                  value={newFolderPath}
                  onChange={(e) => setNewFolderPath(e.target.value)}
                  className="flex-1 text-sm"
                  data-testid="new-folder-input"
                />
                <Button size="sm" onClick={addFolder} data-testid="add-folder-btn">
                  <Plus className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Files Panel */}
        <Card className="col-span-12 md:col-span-9 panel" data-testid="files-panel">
          <CardHeader className="panel-header">
            <CardTitle className="panel-title">
              CSV Files
              {selectedFolder && (
                <span className="text-muted-foreground font-normal ml-2">
                  ({selectedFolder.path})
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {loading ? (
              <div className="flex items-center justify-center h-[300px]">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
              </div>
            ) : files.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-[300px] text-muted-foreground">
                <HardDrive className="w-12 h-12 mb-2 opacity-50" />
                <p>No CSV files found</p>
                <p className="text-sm">Upload a file or select a different folder</p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>Rows</TableHead>
                    <TableHead>Columns</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {files.map((file, index) => (
                    <TableRow key={index} data-testid={`file-row-${index}`}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <FileSpreadsheet className="w-4 h-4 text-green-500" />
                          <span className="font-mono text-sm">{file.name}</span>
                        </div>
                      </TableCell>
                      <TableCell className="font-mono text-sm">
                        {formatFileSize(file.size)}
                      </TableCell>
                      <TableCell className="font-mono text-sm">
                        {file.rows?.toLocaleString() || '-'}
                      </TableCell>
                      <TableCell className="font-mono text-sm">
                        {file.columns?.length || '-'}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => previewFile(file)}
                          data-testid={`preview-btn-${index}`}
                        >
                          <Eye className="w-4 h-4 mr-1" />
                          Preview
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Preview Dialog */}
      <Dialog open={showPreview} onOpenChange={setShowPreview}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden" data-testid="preview-dialog">
          <DialogHeader>
            <DialogTitle className="font-mono">
              {selectedFile?.name}
            </DialogTitle>
          </DialogHeader>
          {previewData && (
            <div className="space-y-4">
              <div className="flex gap-4 text-sm">
                <span className="text-muted-foreground">
                  Total Rows: <span className="text-foreground font-mono">{previewData.total_rows?.toLocaleString()}</span>
                </span>
                <span className="text-muted-foreground">
                  Columns: <span className="text-foreground font-mono">{previewData.columns?.length}</span>
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {previewData.columns?.map((col, i) => (
                  <span
                    key={i}
                    className="px-2 py-1 text-xs font-mono bg-accent rounded"
                  >
                    {col}: <span className="text-muted-foreground">{previewData.dtypes?.[col]}</span>
                  </span>
                ))}
              </div>
              <ScrollArea className="h-[400px] border border-border rounded-md">
                <Table>
                  <TableHeader>
                    <TableRow>
                      {previewData.columns?.map((col, i) => (
                        <TableHead key={i} className="font-mono text-xs whitespace-nowrap">
                          {col}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {previewData.data?.slice(0, 50).map((row, i) => (
                      <TableRow key={i}>
                        {previewData.columns?.map((col, j) => (
                          <TableCell key={j} className="font-mono text-xs whitespace-nowrap">
                            {typeof row[col] === 'number'
                              ? row[col].toFixed?.(4) || row[col]
                              : String(row[col]).slice(0, 20)}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default DataManagement;
