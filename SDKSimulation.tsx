import { useRef, useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Terminal, Trash2, Send, Circle } from 'lucide-react';

interface SDKSimulationProps {
  consoleOutput: string[];
  onClearConsole: () => void;
}

export const SDKSimulation = ({ consoleOutput, onClearConsole }: SDKSimulationProps) => {
  const [command, setCommand] = useState('');
  const [localOutput, setLocalOutput] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const allOutput = [...consoleOutput, ...localOutput];

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [allOutput]);

  const handleCommand = (cmd: string) => {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    const newOutput = [`[${timestamp}] $ ${cmd}`];

    // Simulate robotics API responses
    if (cmd === 'robot.connect()') {
      setIsConnected(true);
      newOutput.push(`[${timestamp}] [INFO] Establishing connection to robot...`);
      newOutput.push(`[${timestamp}] [OK] Connected to robot at 192.168.1.100:5000`);
      newOutput.push(`[${timestamp}] [OK] Firmware: v3.2.1 | Status: READY`);
    } else if (cmd === 'robot.status()') {
      newOutput.push(`[${timestamp}] [INFO] Querying robot status...`);
      newOutput.push(`[${timestamp}] {`);
      newOutput.push(`[${timestamp}]   "state": "IDLE",`);
      newOutput.push(`[${timestamp}]   "battery": 87,`);
      newOutput.push(`[${timestamp}]   "joints": 12,`);
      newOutput.push(`[${timestamp}]   "uptime": "4h 23m"`);
      newOutput.push(`[${timestamp}] }`);
    } else if (cmd.startsWith('robot.move(')) {
      newOutput.push(`[${timestamp}] [INFO] Executing motion command...`);
      newOutput.push(`[${timestamp}] [OK] Motion executed successfully`);
    } else if (cmd.startsWith('robot.get_joint(')) {
      newOutput.push(`[${timestamp}] [INFO] Reading joint state...`);
      newOutput.push(`[${timestamp}] { "angle": 45.2, "velocity": 0.0, "torque": 1.23 }`);
    } else if (cmd === 'robot.calibrate()') {
      newOutput.push(`[${timestamp}] [INFO] Starting calibration sequence...`);
      newOutput.push(`[${timestamp}] [INFO] Calibrating joint 1/12...`);
      newOutput.push(`[${timestamp}] [OK] Calibration complete`);
    } else if (cmd === 'help') {
      newOutput.push(`[${timestamp}] Available commands:`);
      newOutput.push(`[${timestamp}]   robot.connect()      - Connect to robot`);
      newOutput.push(`[${timestamp}]   robot.status()       - Get robot status`);
      newOutput.push(`[${timestamp}]   robot.move(x,y,z)    - Move end effector`);
      newOutput.push(`[${timestamp}]   robot.get_joint(id)  - Read joint state`);
      newOutput.push(`[${timestamp}]   robot.calibrate()    - Run calibration`);
      newOutput.push(`[${timestamp}]   clear                - Clear console`);
    } else if (cmd === 'clear') {
      setLocalOutput([]);
      onClearConsole();
      setCommand('');
      return;
    } else {
      newOutput.push(`[${timestamp}] [ERROR] Unknown command: ${cmd}`);
      newOutput.push(`[${timestamp}] [INFO] Type 'help' for available commands`);
    }

    setLocalOutput(prev => [...prev, ...newOutput]);
    setCommand('');
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (command.trim()) {
      handleCommand(command.trim());
    }
  };

  const handleClear = () => {
    setLocalOutput([]);
    onClearConsole();
  };

  return (
    <Card className="border-border overflow-hidden h-[600px] flex flex-col bg-card">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-muted/50 border-b border-border">
        <div className="flex items-center gap-3">
          <Terminal className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Console</span>
          <div className="flex items-center gap-1.5">
            <Circle className={`h-2 w-2 ${isConnected ? 'fill-green-500 text-green-500' : 'fill-muted-foreground text-muted-foreground'}`} />
            <span className="text-xs text-muted-foreground">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
        <Button size="sm" variant="ghost" onClick={handleClear} className="h-7 text-xs">
          <Trash2 className="h-3 w-3 mr-1.5" />
          Clear
        </Button>
      </div>

      {/* Console Output */}
      <ScrollArea className="flex-1 p-4 bg-background">
        <div ref={scrollRef} className="space-y-0.5 font-mono text-xs">
          {allOutput.length === 0 ? (
            <div className="text-muted-foreground">
              <p>Robotics SDK Console v2.0</p>
              <p className="mt-2">Type 'help' for available commands.</p>
              <p>Type 'robot.connect()' to connect to a robot.</p>
            </div>
          ) : (
            allOutput.map((line, index) => (
              <div 
                key={index}
                className={`leading-relaxed ${
                  line.includes('[ERROR]') ? 'text-red-400' :
                  line.includes('[OK]') ? 'text-green-400' :
                  line.includes('[INFO]') ? 'text-blue-400' :
                  line.includes('$') ? 'text-primary' :
                  'text-foreground/80'
                }`}
              >
                {line}
              </div>
            ))
          )}
        </div>
      </ScrollArea>

      {/* Command Input */}
      <div className="p-3 border-t border-border bg-muted/30">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <div className="flex-1 flex items-center gap-2 bg-background border border-border rounded-lg px-3">
            <span className="text-muted-foreground text-sm">$</span>
            <Input
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="Enter command..."
              className="border-0 bg-transparent focus-visible:ring-0 font-mono text-sm px-0"
            />
          </div>
          <Button type="submit" size="sm" className="px-4">
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </Card>
  );
};
