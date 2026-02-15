import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, PerspectiveCamera, Environment, TransformControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Download, Upload, RotateCcw, Eye, Maximize2, Grid3x3, Sun } from 'lucide-react';
import { useState, Suspense } from 'react';

interface RobotPart {
  id: string;
  type: string;
  position: [number, number, number];
  rotation: [number, number, number];
  color: string;
}

interface RobotDesigner3DProps {
  parts: RobotPart[];
  selectedPart: string | null;
  onPartSelect: (id: string) => void;
  onPartUpdate: (id: string, position: [number, number, number], rotation: [number, number, number]) => void;
}

export const RobotDesigner3D = ({ parts, selectedPart, onPartSelect, onPartUpdate }: RobotDesigner3DProps) => {
  const [transformMode, setTransformMode] = useState<'translate' | 'rotate' | 'scale'>('translate');
  const [showGrid, setShowGrid] = useState(true);
  const [showGizmo, setShowGizmo] = useState(false);
  const [shadowsEnabled, setShadowsEnabled] = useState(true);
  const [wireframeMode, setWireframeMode] = useState(false);
  
  return (
    <Card className="glass-effect border-border/50 h-full">
      <Tabs defaultValue="design" className="w-full">
        <div className="p-4 border-b border-border/50">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="font-semibold">3D Robot Designer</h3>
              <p className="text-xs text-muted-foreground">Design and configure your humanoid robot</p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="text-xs">
                <Eye className="h-3 w-3 mr-1" />
                {parts.length} Parts
              </Badge>
            </div>
          </div>
          
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="design">Design Mode</TabsTrigger>
            <TabsTrigger value="settings">View Settings</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="design" className="mt-0 p-4 border-b border-border/50">
          <div className="flex items-center justify-between">
            <div className="flex gap-1 border border-border rounded-md">
              <Button 
                size="sm" 
                variant={transformMode === 'translate' ? 'default' : 'ghost'}
                onClick={() => setTransformMode('translate')}
                className="text-xs h-8"
              >
                Move
              </Button>
              <Button 
                size="sm" 
                variant={transformMode === 'rotate' ? 'default' : 'ghost'}
                onClick={() => setTransformMode('rotate')}
                className="text-xs h-8"
              >
                Rotate
              </Button>
              <Button 
                size="sm" 
                variant={transformMode === 'scale' ? 'default' : 'ghost'}
                onClick={() => setTransformMode('scale')}
                className="text-xs h-8"
              >
                Scale
              </Button>
            </div>
            
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline">
                <Upload className="h-4 w-4 mr-2" />
                Import
              </Button>
              <Button size="sm" variant="outline">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
              <Button size="sm" variant="outline">
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="settings" className="mt-0 p-4 border-b border-border/50">
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2 text-sm">
                <Grid3x3 className="h-4 w-4" />
                Show Grid
              </Label>
              <Switch checked={showGrid} onCheckedChange={setShowGrid} />
            </div>
            
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2 text-sm">
                <Maximize2 className="h-4 w-4" />
                Gizmo Helper
              </Label>
              <Switch checked={showGizmo} onCheckedChange={setShowGizmo} />
            </div>
            
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2 text-sm">
                <Sun className="h-4 w-4" />
                Shadows
              </Label>
              <Switch checked={shadowsEnabled} onCheckedChange={setShadowsEnabled} />
            </div>
            
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2 text-sm">
                <Grid3x3 className="h-4 w-4" />
                Wireframe
              </Label>
              <Switch checked={wireframeMode} onCheckedChange={setWireframeMode} />
            </div>
          </div>
        </TabsContent>
      </Tabs>
      
      <div className="h-[700px] bg-muted/20">
        <Canvas shadows={shadowsEnabled}>
          <PerspectiveCamera makeDefault position={[3, 2, 5]} />
          <OrbitControls 
            enableDamping
            dampingFactor={0.05}
            minDistance={2}
            maxDistance={10}
            makeDefault
          />
          
          <ambientLight intensity={0.5} />
          <directionalLight 
            position={[10, 10, 5]} 
            intensity={1}
            castShadow={shadowsEnabled}
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />
          <spotLight position={[-10, 10, -5]} intensity={0.3} />
          <hemisphereLight args={['#ffffff', '#444444', 0.3]} />
          
          <Environment preset="studio" />
          
          {parts.map(part => {
            const isSelected = selectedPart === part.id;
            return (
              <group key={part.id}>
                <mesh 
                  position={part.position} 
                  rotation={part.rotation}
                  onClick={() => onPartSelect(part.id)}
                  castShadow
                  receiveShadow
                >
                  {part.type === 'torso' && <boxGeometry args={[1, 1.5, 0.5]} />}
                  {part.type === 'head' && <sphereGeometry args={[0.4, 32, 32]} />}
                  {(part.type === 'upper-arm' || part.type === 'upper-leg') && (
                    <cylinderGeometry args={[0.15, 0.15, 0.8, 16]} />
                  )}
                  {(part.type === 'lower-arm' || part.type === 'lower-leg') && (
                    <cylinderGeometry args={[0.12, 0.12, 0.7, 16]} />
                  )}
                  {(part.type === 'hand' || part.type === 'foot') && (
                    <boxGeometry args={[0.2, 0.1, 0.3]} />
                  )}
                  {!['torso', 'head', 'upper-arm', 'upper-leg', 'lower-arm', 'lower-leg', 'hand', 'foot'].includes(part.type) && (
                    <boxGeometry args={[0.3, 0.3, 0.3]} />
                  )}
                  <meshStandardMaterial 
                    color={isSelected ? '#3b82f6' : part.color}
                    metalness={0.3}
                    roughness={0.4}
                    wireframe={wireframeMode}
                  />
                </mesh>
                
                {isSelected && (
                  <TransformControls
                    mode={transformMode}
                    onObjectChange={(e) => {
                      const target = (e as any)?.target;
                      if (target?.object) {
                        const pos = target.object.position;
                        const rot = target.object.rotation;
                        onPartUpdate(
                          part.id,
                          [pos.x, pos.y, pos.z],
                          [rot.x, rot.y, rot.z]
                        );
                      }
                    }}
                  >
                    <mesh position={part.position} rotation={part.rotation}>
                      <boxGeometry args={[0.01, 0.01, 0.01]} />
                      <meshBasicMaterial visible={false} />
                    </mesh>
                  </TransformControls>
                )}
              </group>
            );
          })}
          
          {showGrid && (
            <Grid 
              args={[20, 20]} 
              cellSize={0.5}
              cellThickness={0.5}
              cellColor="#6b7280"
              sectionSize={2}
              sectionThickness={1}
              sectionColor="#9ca3af"
              fadeDistance={30}
              fadeStrength={1}
              followCamera={false}
              infiniteGrid={true}
            />
          )}
          
          {showGizmo && (
            <Suspense fallback={null}>
              <GizmoHelper 
                alignment="bottom-right" 
                margin={[80, 80] as [number, number]}
              >
                <GizmoViewport axisColors={['#ff0000', '#00ff00', '#0000ff']} labelColor="white" />
              </GizmoHelper>
            </Suspense>
          )}
        </Canvas>
      </div>
      
      <div className="p-3 border-t border-border/50 flex items-center justify-between text-xs text-muted-foreground">
        <span>{parts.length} parts configured</span>
        <span>Left click + drag to rotate • Scroll to zoom • Right click + drag to pan</span>
      </div>
    </Card>
  );
};
