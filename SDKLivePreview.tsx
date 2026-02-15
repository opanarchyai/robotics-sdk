import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Text } from '@react-three/drei';
import * as THREE from 'three';

export type PreviewMode = 'control' | 'perception' | 'hardware' | 'simulation' | 'ai' | 'communication' | 'safety';

interface RobotSimState {
  jointAngles: number[];
  gripperAperture: number;
  lidarRays: { angle: number; distance: number; hit: boolean }[];
  showLidar: boolean;
  previewMode: PreviewMode;
}

interface SDKLivePreviewProps {
  robotState: RobotSimState;
}

const MODE_LABELS: Record<PreviewMode, string> = {
  control: 'Control Systems',
  perception: 'Sensor Suite',
  hardware: 'Hardware Components',
  simulation: 'Full Simulation',
  ai: 'AI / Training',
  communication: 'Communication',
  safety: 'Safety Monitor',
};

const MODE_COLORS: Record<PreviewMode, string> = {
  control: '#3498db',
  perception: '#e74c3c',
  hardware: '#e67e22',
  simulation: '#2ecc71',
  ai: '#9b59b6',
  communication: '#1abc9c',
  safety: '#f39c12',
};

/* ──── Shared components ──── */

function Joint({ radius = 0.06, color = '#e74c3c' }: { radius?: number; color?: string }) {
  return (
    <mesh>
      <sphereGeometry args={[radius, 16, 16]} />
      <meshStandardMaterial color={color} metalness={0.5} roughness={0.4} />
    </mesh>
  );
}

function Link({ length, radius = 0.04, color }: { length: number; radius?: number; color: string }) {
  return (
    <mesh position={[length / 2, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
      <cylinderGeometry args={[radius, radius, length, 16]} />
      <meshStandardMaterial color={color} metalness={0.6} roughness={0.3} />
    </mesh>
  );
}

function Gripper({ aperture }: { aperture: number }) {
  const hw = aperture * 0.04;
  return (
    <group>
      <group position={[0, hw, 0]}>
        <mesh position={[0.08, 0, 0]}>
          <boxGeometry args={[0.16, 0.015, 0.03]} />
          <meshStandardMaterial color="#95a5a6" metalness={0.7} roughness={0.2} />
        </mesh>
      </group>
      <group position={[0, -hw, 0]}>
        <mesh position={[0.08, 0, 0]}>
          <boxGeometry args={[0.16, 0.015, 0.03]} />
          <meshStandardMaterial color="#95a5a6" metalness={0.7} roughness={0.2} />
        </mesh>
      </group>
      <mesh position={[-0.02, 0, 0]}>
        <boxGeometry args={[0.04, 0.1, 0.04]} />
        <meshStandardMaterial color="#7f8c8d" metalness={0.6} roughness={0.3} />
      </mesh>
    </group>
  );
}

function RobotArm({ jointAngles, gripperAperture, accentColor = '#3498db' }: { jointAngles: number[]; gripperAperture: number; accentColor?: string }) {
  const base = jointAngles[0] || 0;
  const shoulder = jointAngles[1] || 0;
  const elbow = jointAngles[2] || 0;

  return (
    <group>
      <mesh position={[0, 0.15, 0]}>
        <cylinderGeometry args={[0.12, 0.15, 0.3, 32]} />
        <meshStandardMaterial color="#2c3e50" metalness={0.7} roughness={0.2} />
      </mesh>
      <group position={[0, 0.3, 0]} rotation={[0, base, 0]}>
        <Joint radius={0.07} />
        <group rotation={[0, 0, shoulder + Math.PI / 4]}>
          <Link length={1.0} color={accentColor} />
          <group position={[1.0, 0, 0]}>
            <Joint />
            <group rotation={[0, 0, elbow - Math.PI / 4]}>
              <Link length={0.8} radius={0.035} color="#2ecc71" />
              <group position={[0.8, 0, 0]}>
                <Joint radius={0.04} />
                <Gripper aperture={gripperAperture} />
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

function Floor() {
  return (
    <>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#1a1a2e" transparent opacity={0.8} />
      </mesh>
      <Grid position={[0, 0, 0]} args={[20, 20]} cellSize={0.5} cellThickness={0.5} cellColor="#334155" sectionSize={2} sectionThickness={1} sectionColor="#475569" fadeDistance={15} infiniteGrid />
    </>
  );
}

/* ──── Mode-specific scenes ──── */

/** Control: robot arm with trajectory trail and IK target */
function ControlScene({ jointAngles, gripperAperture }: { jointAngles: number[]; gripperAperture: number }) {
  const a1 = jointAngles[1] || 0;
  const a2 = jointAngles[2] || 0;
  const ex = Math.cos(a1 + Math.PI / 4) + 0.8 * Math.cos(a1 + a2);
  const ey = 0.3 + Math.sin(a1 + Math.PI / 4) + 0.8 * Math.sin(a1 + a2);

  return (
    <group>
      <RobotArm jointAngles={jointAngles} gripperAperture={gripperAperture} accentColor="#3498db" />
      {/* IK target marker */}
      <mesh position={[ex, ey, 0]}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial color="#f1c40f" transparent opacity={0.8} />
      </mesh>
      {/* Trajectory arc hint */}
      <mesh position={[ex, ey, 0]}>
        <ringGeometry args={[0.08, 0.1, 32]} />
        <meshBasicMaterial color="#f1c40f" transparent opacity={0.3} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

/** Perception: robot with LiDAR rays and sensor cones */
function PerceptionScene({ jointAngles, gripperAperture, lidarRays }: { jointAngles: number[]; gripperAperture: number; lidarRays: { angle: number; distance: number; hit: boolean }[] }) {
  const rayGeo = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    for (const ray of lidarRays) {
      const dx = Math.cos(ray.angle) * ray.distance;
      const dz = Math.sin(ray.angle) * ray.distance;
      positions.push(0, 0.5, 0, dx, 0.5, dz);
      if (ray.hit) colors.push(1, 0.2, 0.2, 1, 0.2, 0.2);
      else colors.push(0.2, 1, 0.2, 0.2, 1, 0.2);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    return geo;
  }, [lidarRays]);

  return (
    <group>
      <RobotArm jointAngles={jointAngles} gripperAperture={gripperAperture} accentColor="#e74c3c" />
      {/* LiDAR rays */}
      {lidarRays.length > 0 && (
        <lineSegments geometry={rayGeo}>
          <lineBasicMaterial vertexColors transparent opacity={0.5} />
        </lineSegments>
      )}
      {/* Camera frustum on wrist */}
      <group position={[0, 1.2, 0]} rotation={[0, 0, -Math.PI / 6]}>
        <mesh>
          <coneGeometry args={[0.3, 0.8, 4]} />
          <meshBasicMaterial color="#e74c3c" transparent opacity={0.08} wireframe />
        </mesh>
      </group>
      {/* Obstacles */}
      <mesh position={[2, 0.5, 1]}>
        <boxGeometry args={[0.5, 1, 0.5]} />
        <meshStandardMaterial color="#e67e22" transparent opacity={0.5} />
      </mesh>
      <mesh position={[-1.5, 0.4, -2]}>
        <cylinderGeometry args={[0.3, 0.3, 0.8, 16]} />
        <meshStandardMaterial color="#9b59b6" transparent opacity={0.5} />
      </mesh>
      <mesh position={[1, 0.3, -1.5]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshStandardMaterial color="#1abc9c" transparent opacity={0.5} />
      </mesh>
    </group>
  );
}

/** Hardware: exploded view with servo details, gears, and labels */
function HardwareScene({ jointAngles, gripperAperture }: { jointAngles: number[]; gripperAperture: number }) {
  return (
    <group>
      {/* Base servo assembly */}
      <group position={[0, 0.15, 0]}>
        <mesh>
          <cylinderGeometry args={[0.12, 0.15, 0.3, 32]} />
          <meshStandardMaterial color="#2c3e50" metalness={0.7} roughness={0.2} />
        </mesh>
        {/* Motor body */}
        <mesh position={[0.18, 0, 0]}>
          <boxGeometry args={[0.08, 0.15, 0.08]} />
          <meshStandardMaterial color="#e67e22" metalness={0.5} roughness={0.3} />
        </mesh>
        {/* Gear */}
        <mesh position={[0, 0.16, 0]} rotation={[0, jointAngles[0] * 3, 0]}>
          <torusGeometry args={[0.06, 0.015, 8, 12]} />
          <meshStandardMaterial color="#f39c12" metalness={0.8} roughness={0.2} />
        </mesh>
      </group>

      {/* Shoulder servo */}
      <group position={[0, 0.45, 0]}>
        <Joint radius={0.07} color="#e67e22" />
        {/* Motor housing */}
        <mesh position={[0.15, 0, 0]}>
          <boxGeometry args={[0.1, 0.12, 0.08]} />
          <meshStandardMaterial color="#e67e22" metalness={0.5} roughness={0.3} />
        </mesh>
        {/* Gear */}
        <mesh rotation={[0, 0, jointAngles[1] * 3]}>
          <torusGeometry args={[0.08, 0.012, 8, 16]} />
          <meshStandardMaterial color="#f39c12" metalness={0.8} roughness={0.2} />
        </mesh>
        {/* Arm link */}
        <group rotation={[0, 0, jointAngles[1] || 0]}>
          <Link length={0.9} radius={0.035} color="#3498db" />
          
          {/* Elbow servo */}
          <group position={[0.9, 0, 0]}>
            <Joint radius={0.06} color="#e67e22" />
            <mesh position={[0.12, 0, 0]}>
              <boxGeometry args={[0.08, 0.1, 0.07]} />
              <meshStandardMaterial color="#e67e22" metalness={0.5} roughness={0.3} />
            </mesh>
            <mesh rotation={[0, 0, jointAngles[2] * 3]}>
              <torusGeometry args={[0.06, 0.01, 8, 12]} />
              <meshStandardMaterial color="#f39c12" metalness={0.8} roughness={0.2} />
            </mesh>
            <group rotation={[0, 0, jointAngles[2] || 0]}>
              <Link length={0.7} radius={0.03} color="#2ecc71" />
              <group position={[0.7, 0, 0]}>
                <Gripper aperture={gripperAperture} />
              </group>
            </group>
          </group>
        </group>
      </group>

      {/* Wiring */}
      {[0.2, 0.5, 0.8].map((y, i) => (
        <mesh key={i} position={[0.2, y, 0]}>
          <cylinderGeometry args={[0.005, 0.005, 0.15, 8]} />
          <meshBasicMaterial color={['#e74c3c', '#f1c40f', '#2ecc71'][i]} />
        </mesh>
      ))}
    </group>
  );
}

/** AI: robot with reward field visualization */
function AIScene({ jointAngles, gripperAperture }: { jointAngles: number[]; gripperAperture: number }) {
  const ref = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y += delta * 0.1;
  });

  return (
    <group>
      <RobotArm jointAngles={jointAngles} gripperAperture={gripperAperture} accentColor="#9b59b6" />
      {/* Reward field grid */}
      <group ref={ref} position={[0, 0.01, 0]}>
        {Array.from({ length: 8 }).map((_, i) =>
          Array.from({ length: 8 }).map((_, j) => {
            const x = (i - 3.5) * 0.4;
            const z = (j - 3.5) * 0.4;
            const dist = Math.sqrt(x * x + z * z);
            const reward = Math.max(0, 1 - dist / 2);
            return (
              <mesh key={`${i}-${j}`} position={[x, reward * 0.3, z]}>
                <boxGeometry args={[0.35, 0.02, 0.35]} />
                <meshStandardMaterial color={`hsl(${reward * 120}, 70%, 50%)`} transparent opacity={0.3 + reward * 0.5} />
              </mesh>
            );
          })
        )}
      </group>
      {/* Target sphere */}
      <mesh position={[1.0, 0.8, 0]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshBasicMaterial color="#f1c40f" transparent opacity={0.7} />
      </mesh>
    </group>
  );
}

/** Communication: robot with data flow visualization */
function CommunicationScene({ jointAngles, gripperAperture }: { jointAngles: number[]; gripperAperture: number }) {
  const particlesRef = useRef<THREE.Points>(null);
  
  const particleGeo = useMemo(() => {
    const positions = new Float32Array(60 * 3);
    for (let i = 0; i < 60; i++) {
      const angle = (i / 60) * Math.PI * 2;
      const radius = 1.5 + Math.sin(i * 0.5) * 0.3;
      positions[i * 3] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = 0.5 + (i / 60) * 1.5;
      positions[i * 3 + 2] = Math.sin(angle) * radius;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    return geo;
  }, []);

  useFrame((_, delta) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y += delta * 0.5;
    }
  });

  return (
    <group>
      <RobotArm jointAngles={jointAngles} gripperAperture={gripperAperture} accentColor="#1abc9c" />
      {/* Data flow particles */}
      <points ref={particlesRef} geometry={particleGeo}>
        <pointsMaterial color="#1abc9c" size={0.04} transparent opacity={0.6} />
      </points>
      {/* Signal rings */}
      {[0.8, 1.2, 1.6].map((r, i) => (
        <mesh key={i} position={[0, 1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[r - 0.02, r, 32]} />
          <meshBasicMaterial color="#1abc9c" transparent opacity={0.15 - i * 0.04} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </group>
  );
}

/** Safety: robot with limit zones and warning indicators */
function SafetyScene({ jointAngles, gripperAperture }: { jointAngles: number[]; gripperAperture: number }) {
  const a1 = jointAngles[1] || 0;
  const inSoftLimit = Math.abs(a1) > Math.PI * 0.7;
  const inHardLimit = Math.abs(a1) > Math.PI * 0.9;

  return (
    <group>
      <RobotArm jointAngles={jointAngles} gripperAperture={gripperAperture} accentColor={inHardLimit ? '#e74c3c' : inSoftLimit ? '#f39c12' : '#2ecc71'} />
      {/* Safe zone (green) */}
      <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.3, 1.5, 32]} />
        <meshBasicMaterial color="#2ecc71" transparent opacity={0.08} side={THREE.DoubleSide} />
      </mesh>
      {/* Warning zone (yellow) */}
      <mesh position={[0, 0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.5, 1.8, 32]} />
        <meshBasicMaterial color="#f39c12" transparent opacity={0.1} side={THREE.DoubleSide} />
      </mesh>
      {/* Danger zone (red) */}
      <mesh position={[0, 0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.8, 2.2, 32]} />
        <meshBasicMaterial color="#e74c3c" transparent opacity={0.1} side={THREE.DoubleSide} />
      </mesh>
      {/* Collision boundary sphere */}
      <mesh position={[0, 0.8, 0]}>
        <sphereGeometry args={[1.85, 32, 32]} />
        <meshBasicMaterial color={inHardLimit ? '#e74c3c' : '#2ecc71'} transparent opacity={0.04} wireframe />
      </mesh>
    </group>
  );
}

/* ──── Main Component ──── */

export const SDKLivePreview = ({ robotState }: SDKLivePreviewProps) => {
  const mode = robotState.previewMode || 'simulation';
  const accentColor = MODE_COLORS[mode];

  return (
    <div className="h-full w-full bg-[#0a0a1a] rounded-lg overflow-hidden relative">
      <Canvas camera={{ position: [3, 2.5, 3], fov: 50 }} shadows gl={{ antialias: true }}>
        <color attach="background" args={['#0d1117']} />
        <fog attach="fog" args={['#0d1117', 8, 20]} />

        <ambientLight intensity={0.3} />
        <directionalLight position={[5, 8, 5]} intensity={1} castShadow />
        <pointLight position={[-3, 3, -3]} intensity={0.5} color={accentColor} />
        <pointLight position={[3, 2, -2]} intensity={0.3} color="#2ecc71" />

        {mode === 'control' && <ControlScene jointAngles={robotState.jointAngles} gripperAperture={robotState.gripperAperture} />}
        {mode === 'perception' && <PerceptionScene jointAngles={robotState.jointAngles} gripperAperture={robotState.gripperAperture} lidarRays={robotState.lidarRays} />}
        {mode === 'hardware' && <HardwareScene jointAngles={robotState.jointAngles} gripperAperture={robotState.gripperAperture} />}
        {mode === 'simulation' && <RobotArm jointAngles={robotState.jointAngles} gripperAperture={robotState.gripperAperture} />}
        {mode === 'ai' && <AIScene jointAngles={robotState.jointAngles} gripperAperture={robotState.gripperAperture} />}
        {mode === 'communication' && <CommunicationScene jointAngles={robotState.jointAngles} gripperAperture={robotState.gripperAperture} />}
        {mode === 'safety' && <SafetyScene jointAngles={robotState.jointAngles} gripperAperture={robotState.gripperAperture} />}

        <Floor />
        <OrbitControls enableDamping dampingFactor={0.05} minDistance={1} maxDistance={15} target={[0, 0.5, 0]} />
      </Canvas>

      {/* Mode indicator */}
      <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-background/80 backdrop-blur-sm border border-border rounded-md px-2 py-1">
        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: accentColor }} />
        <span className="text-[10px] font-medium text-foreground">{MODE_LABELS[mode]}</span>
      </div>

      {/* Telemetry */}
      <div className="absolute bottom-2 left-2 text-[10px] font-mono text-muted-foreground/60 space-y-0.5">
        <div>Joints: {robotState.jointAngles.map(a => `${(a * 180 / Math.PI).toFixed(1)}°`).join(' | ')}</div>
        <div>Gripper: {(robotState.gripperAperture * 100).toFixed(0)}%</div>
        {robotState.showLidar && <div>LiDAR: {robotState.lidarRays.length} rays</div>}
      </div>
    </div>
  );
};
