import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Plus, Package } from 'lucide-react';
import { motion } from 'framer-motion';

interface PartTemplate {
  id: string;
  name: string;
  type: string;
  category: string;
  description: string;
  specs: string;
}

interface RobotPartLibraryProps {
  onAddPart: (type: string) => void;
}

export const RobotPartLibrary = ({ onAddPart }: RobotPartLibraryProps) => {
  const partTemplates: PartTemplate[] = [
    {
      id: 'torso-v1',
      name: 'Core Torso',
      type: 'torso',
      category: 'Structure',
      description: 'Main chassis with motor mounts',
      specs: 'Weight: 2.5kg'
    },
    {
      id: 'head-v1',
      name: 'Sensor Head',
      type: 'head',
      category: 'Sensors',
      description: 'Multi-sensor array housing',
      specs: 'Cameras: 2x RGB'
    },
    {
      id: 'upper-arm-v1',
      name: 'Upper Arm',
      type: 'upper-arm',
      category: 'Limbs',
      description: 'Shoulder to elbow segment',
      specs: 'DOF: 2, Torque: 50Nm'
    },
    {
      id: 'lower-arm-v1',
      name: 'Lower Arm',
      type: 'lower-arm',
      category: 'Limbs',
      description: 'Elbow to wrist segment',
      specs: 'DOF: 1, Torque: 30Nm'
    },
    {
      id: 'hand-v1',
      name: 'Gripper Hand',
      type: 'hand',
      category: 'End Effectors',
      description: '5-finger articulated gripper',
      specs: 'Grip Force: 100N'
    },
    {
      id: 'upper-leg-v1',
      name: 'Upper Leg',
      type: 'upper-leg',
      category: 'Limbs',
      description: 'Hip to knee segment',
      specs: 'DOF: 3, Torque: 80Nm'
    },
    {
      id: 'lower-leg-v1',
      name: 'Lower Leg',
      type: 'lower-leg',
      category: 'Limbs',
      description: 'Knee to ankle segment',
      specs: 'DOF: 1, Torque: 60Nm'
    },
    {
      id: 'foot-v1',
      name: 'Stabilizer Foot',
      type: 'foot',
      category: 'End Effectors',
      description: 'Balance platform with sensors',
      specs: 'Force Sensors: 4'
    }
  ];

  const categories = Array.from(new Set(partTemplates.map(p => p.category)));

  return (
    <Card className="glass-effect border-border/50 h-full">
      <CardHeader className="border-b border-border/50">
        <CardTitle className="flex items-center gap-2 text-base">
          <Package className="h-5 w-5 text-primary" />
          Part Library
        </CardTitle>
        <p className="text-xs text-muted-foreground">Add components to build your robot</p>
      </CardHeader>
      
      <ScrollArea className="h-[calc(600px-120px)]">
        <CardContent className="space-y-4 p-4">
          {categories.map(category => (
            <div key={category} className="space-y-2">
              <h4 className="text-sm font-semibold text-muted-foreground">{category}</h4>
              <div className="space-y-2">
                {partTemplates
                  .filter(part => part.category === category)
                  .map(part => (
                    <motion.div
                      key={part.id}
                      whileHover={{ scale: 1.02 }}
                      className="p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-all border border-border/50"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <h5 className="font-medium text-sm">{part.name}</h5>
                          <p className="text-xs text-muted-foreground mt-1">
                            {part.description}
                          </p>
                        </div>
                        <Button 
                          size="sm" 
                          variant="ghost"
                          onClick={() => onAddPart(part.type)}
                          className="h-8 w-8 p-0"
                        >
                          <Plus className="h-4 w-4" />
                        </Button>
                      </div>
                      <Badge variant="secondary" className="text-xs">
                        {part.specs}
                      </Badge>
                    </motion.div>
                  ))}
              </div>
            </div>
          ))}
        </CardContent>
      </ScrollArea>
    </Card>
  );
};
