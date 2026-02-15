import Editor from '@monaco-editor/react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Play, Copy, Download, RotateCcw } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface SDKCodeEditorProps {
  code: string;
  language: string;
  filename: string;
  onChange: (code: string) => void;
  onRun: () => void;
  moduleId?: string;
}

export const SDKCodeEditor = ({ code, language, filename, onChange, onRun }: SDKCodeEditorProps) => {
  const { toast } = useToast();

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    toast({ title: "Copied to clipboard" });
  };

  const handleDownload = () => {
    const ext = language === 'python' ? 'py' : language === 'cpp' ? 'cpp' : 'ts';
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename.replace(/\s+/g, '_').toLowerCase()}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-3 py-1 bg-muted/20 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-red-500/60" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/60" />
            <div className="w-3 h-3 rounded-full bg-emerald-500/60" />
          </div>
          <span className="text-[11px] font-medium text-foreground/80 ml-2 truncate max-w-[200px]">{filename}</span>
          <Badge variant="secondary" className="text-[10px] font-mono h-4 px-1.5">{language}</Badge>
        </div>
        <div className="flex items-center gap-0.5">
          <Button size="sm" variant="ghost" onClick={handleCopy} className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground">
            <Copy className="h-3 w-3" />
          </Button>
          <Button size="sm" variant="ghost" onClick={handleDownload} className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground">
            <Download className="h-3 w-3" />
          </Button>
          <Button size="sm" onClick={onRun} className="h-6 px-3 text-[11px] ml-1 bg-emerald-600 hover:bg-emerald-500 text-white">
            <Play className="h-3 w-3 mr-1" />Run
          </Button>
        </div>
      </div>

      <div className="flex-1">
        <Editor
          height="100%"
          language={language}
          value={code}
          onChange={(v) => onChange(v || '')}
          theme="vs-dark"
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            lineNumbers: 'on',
            scrollBeyondLastLine: false,
            automaticLayout: true,
            wordWrap: 'on',
            fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
            fontLigatures: true,
            cursorBlinking: 'smooth',
            cursorSmoothCaretAnimation: 'on',
            smoothScrolling: true,
            padding: { top: 12, bottom: 12 },
            bracketPairColorization: { enabled: true },
            renderLineHighlight: 'line',
            lineHeight: 20,
            letterSpacing: 0.3,
            scrollbar: { vertical: 'auto', horizontal: 'auto', verticalScrollbarSize: 6, horizontalScrollbarSize: 6 },
            overviewRulerBorder: false,
            hideCursorInOverviewRuler: true,
            guides: { indentation: true, bracketPairs: true },
          }}
        />
      </div>
    </div>
  );
};
