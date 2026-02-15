import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Settings, Trash2, Lock, Unlock } from 'lucide-react';
import { useState } from 'react';

interface Joint {
  id: string;
  name: string;
  type: string;
  range: [number, number];
  value: number;
  locked: boolean;
  torqueLimit: number;
  speed: number;
}

interface RobotJointConfigProps {
  selectedPartId: string | null;
  onDeletePart: (id: string) => void;
}

export const RobotJointConfig = ({ selectedPartId, onDeletePart }: RobotJointConfigProps) => {
  const [joints, setJoints] = useState<Joint[]>([
    {
      id: 'joint-1',
      name: 'Shoulder Pitch',
      type: 'revolute',
      range: [-90, 90],
      value: 0,
      locked: false,
      torqueLimit: 50,
      speed: 45
    },
    {
      id: 'joint-2',
      name: 'Shoulder Roll',
      type: 'revolute',
      range: [-45, 180],
      value: 0,
      locked: false,
      torqueLimit: 50,
      speed: 45
    },
    {
      id: 'joint-3',
      name: 'Elbow',
      type: 'revolute',
      range: [0, 150],
      value: 30,
      locked: false,
      torqueLimit: 30,
      speed: 60
    }
  ]);

  const toggleLock = (jointId: string) => {
    setJoints(prev => prev.map(j => 
      j.id === jointId ? { ...j, locked: !j.locked } : j
    ));
  };

  const updateJointValue = (jointId: string, value: number[]) => {
    setJoints(prev => prev.map(j => 
      j.id === jointId ? { ...j, value: value[0] } : j
    ));
  };

  return (
    <Card className="glass-effect border-border/50 h-full">
      <CardHeader className="border-b border-border/50">
        <CardTitle className="flex items-center gap-2 text-base">
          <Settings className="h-5 w-5 text-primary" />
          Joint Configuration
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          {selectedPartId ? 'Configure selected part' : 'Select a part to configure'}
        </p>
      </CardHeader>
      
      <ScrollArea className="h-[calc(600px-120px)]">
        <CardContent className="space-y-4 p-4">
          {!selectedPartId ? (
            <div className="text-center py-12 text-muted-foreground text-sm">
              <Settings className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>Click on a robot part in the 3D view to configure its joints</p>
            </div>
          ) : (
            <>
              <div className="p-3 rounded-lg bg-primary/10 border border-primary/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-sm">Selected Part</span>
                  <Button 
                    size="sm" 
                    variant="ghost"
                    onClick={() => selectedPartId && onDeletePart(selectedPartId)}
                    className="h-8 text-destructive hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
                <Badge variant="secondary" className="text-xs">
                  {selectedPartId}
                </Badge>
              </div>

              <div className="space-y-4">
                <h4 className="text-sm font-semibold">Joint Controls</h4>
                {joints.map(joint => (
                  <div key={joint.id} className="p-3 rounded-lg bg-muted/30 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm">{joint.name}</span>
                        <Badge variant="outline" className="text-xs">
                          {joint.type}
                        </Badge>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => toggleLock(joint.id)}
                        className="h-7 w-7 p-0"
                      >
                        {joint.locked ? (
                          <Lock className="h-3 w-3" />
                        ) : (
                          <Unlock className="h-3 w-3" />
                        )}
                      </Button>
                    </div>

                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-muted-foreground">Angle</span>
                        <span className="text-xs font-mono">{joint.value}°</span>
                      </div>
                      <Slider
                        value={[joint.value]}
                        onValueChange={(val) => updateJointValue(joint.id, val)}
                        min={joint.range[0]}
                        max={joint.range[1]}
                        step={1}
                        disabled={joint.locked}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground mt-1">
                        <span>{joint.range[0]}°</span>
                        <span>{joint.range[1]}°</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="p-2 rounded bg-background/50">
                        <span className="text-muted-foreground">Torque Limit</span>
                        <p className="font-semibold">{joint.torqueLimit}Nm</p>
                      </div>
                      <div className="p-2 rounded bg-background/50">
                        <span className="text-muted-foreground">Max Speed</span>
                        <p className="font-semibold">{joint.speed}°/s</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="space-y-2">
                <h4 className="text-sm font-semibold">Motion Sequences</h4>
                <Button variant="outline" className="w-full justify-start" size="sm">
                  Record Motion Path
                </Button>
                <Button variant="outline" className="w-full justify-start" size="sm">
                  Test Kinematics
                </Button>
                <Button variant="outline" className="w-full justify-start" size="sm">
                  Export Configuration
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </ScrollArea>
    </Card>
  );
};
