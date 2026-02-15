import { useState, useMemo } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { 
  Cpu, Eye, Wrench, Box, Brain, Radio, ShieldCheck,
  ChevronRight, ChevronDown, FileCode, Search, Sparkles
} from 'lucide-react';

export interface SDKModule {
  id: string;
  name: string;
  description: string;
  category: string;
  categoryId: string;
  language: string;
  code: string;
  tags: string[];
}

const SDK_MODULES: Record<string, SDKModule[]> = {
  control: [
    {
      id: 'pid-controller', name: 'PID Controller',
      description: 'Proportional-integral-derivative controller with anti-windup',
      category: 'Control', categoryId: 'control', language: 'typescript',
      tags: ['pid', 'control-loop', 'servo'],
      code: `// PID Controller — real control loop with anti-windup
// Drives a servo to a target angle with tunable gains

const pid = { kp: 8.0, ki: 0.5, kd: 1.0, integral: 0, prevError: 0 };

function pidUpdate(current, target, dt) {
  const error = target - current;
  pid.integral += error * dt;
  pid.integral = Math.max(-100, Math.min(100, pid.integral)); // Anti-windup
  const derivative = dt > 0 ? (error - pid.prevError) / dt : 0;
  pid.prevError = error;
  return pid.kp * error + pid.ki * pid.integral + pid.kd * derivative;
}

const targetAngle = Math.PI / 2; // 90°
let angle = 0;
const dt = 0.016;

for (let i = 0; i < 150; i++) {
  const correction = pidUpdate(angle, targetAngle, dt);
  const velocity = Math.max(-6.28, Math.min(6.28, correction));
  angle += velocity * dt;
  angle = Math.max(-Math.PI, Math.min(Math.PI, angle));
  robot.setJoint(1, angle);
}

console.log("Target: 90°");
console.log("Final angle: " + (angle * 180 / Math.PI).toFixed(1) + "°");
console.log("Steady-state error: " + ((targetAngle - angle) * 180 / Math.PI).toFixed(3) + "°");
`,
    },
    {
      id: 'inverse-kinematics', name: 'Inverse Kinematics',
      description: 'Analytical 2-link IK solver for planar robot arms',
      category: 'Control', categoryId: 'control', language: 'typescript',
      tags: ['ik', 'kinematics'],
      code: `// 2-Link Inverse Kinematics — analytical closed-form solver
// Computes shoulder & elbow angles to reach a target (x, y)

function solve2LinkIK(tx, ty, l1, l2) {
  const distSq = tx * tx + ty * ty;
  const dist = Math.sqrt(distSq);
  if (dist > l1 + l2) {
    const s = (l1 + l2 - 0.001) / dist;
    return solve2LinkIK(tx * s, ty * s, l1, l2);
  }
  const cosA2 = (distSq - l1*l1 - l2*l2) / (2*l1*l2);
  const a2 = -Math.acos(Math.max(-1, Math.min(1, cosA2)));
  const k1 = l1 + l2 * Math.cos(a2);
  const k2 = l2 * Math.sin(a2);
  const a1 = Math.atan2(ty, tx) - Math.atan2(k2, k1);
  return { shoulder: a1, elbow: a2 };
}

const target = { x: 1.2, y: 0.6 };
const ik = solve2LinkIK(target.x, target.y, 1.0, 0.8);

console.log("Target: (" + target.x + ", " + target.y + ")");
console.log("Shoulder: " + (ik.shoulder * 180/Math.PI).toFixed(1) + "°");
console.log("Elbow: " + (ik.elbow * 180/Math.PI).toFixed(1) + "°");

// Verify with forward kinematics
const ex = Math.cos(ik.shoulder) + 0.8*Math.cos(ik.shoulder+ik.elbow);
const ey = Math.sin(ik.shoulder) + 0.8*Math.sin(ik.shoulder+ik.elbow);
console.log("FK check: (" + ex.toFixed(3) + ", " + ey.toFixed(3) + ")");

robot.setJoint(1, ik.shoulder);
robot.setJoint(2, ik.elbow);
`,
    },
    {
      id: 'trajectory-planner', name: 'Trajectory Planner',
      description: 'Cubic polynomial trajectory with smooth acceleration',
      category: 'Control', categoryId: 'control', language: 'typescript',
      tags: ['trajectory', 'motion', 'spline'],
      code: `// Cubic Trajectory Planner — smooth joint-space motion
// Zero velocity at start and end for jerk-free motion

function cubicTrajectory(q0, qf, tf, steps) {
  const a0 = q0, a1 = 0;
  const a2 = 3*(qf-q0)/(tf*tf);
  const a3 = -2*(qf-q0)/(tf*tf*tf);
  const points = [];
  for (let i = 0; i <= steps; i++) {
    const t = (i/steps)*tf;
    points.push({
      t,
      pos: a0 + a1*t + a2*t*t + a3*t*t*t,
      vel: a1 + 2*a2*t + 3*a3*t*t,
    });
  }
  return points;
}

const shoulderTraj = cubicTrajectory(0, Math.PI/3, 2.0, 120);
const elbowTraj = cubicTrajectory(0, -Math.PI/4, 2.0, 120);

for (let i = 0; i < shoulderTraj.length; i++) {
  robot.setJoint(1, shoulderTraj[i].pos);
  robot.setJoint(2, elbowTraj[i].pos);
}

const maxVel = Math.max(...shoulderTraj.map(p => Math.abs(p.vel)));
console.log("Trajectory points: " + shoulderTraj.length);
console.log("Peak velocity: " + maxVel.toFixed(3) + " rad/s");
console.log("Duration: 2.0s");
`,
    },
    {
      id: 'fabrik-solver', name: 'FABRIK Chain Solver',
      description: 'Forward And Backward Reaching IK for N-link chains',
      category: 'Control', categoryId: 'control', language: 'typescript',
      tags: ['fabrik', 'ik', 'chain'],
      code: `// FABRIK — iterative IK for multi-link chains
// Works by alternating forward and backward passes

function fabrik(joints, target, tolerance, maxIter) {
  const n = joints.length;
  const lengths = [];
  for (let i = 0; i < n-1; i++) {
    const dx = joints[i+1].x - joints[i].x;
    const dy = joints[i+1].y - joints[i].y;
    lengths.push(Math.sqrt(dx*dx + dy*dy));
  }
  const base = {...joints[0]};

  for (let iter = 0; iter < maxIter; iter++) {
    joints[n-1] = {...target};
    for (let i = n-2; i >= 0; i--) {
      const dx = joints[i].x - joints[i+1].x;
      const dy = joints[i].y - joints[i+1].y;
      const d = Math.sqrt(dx*dx + dy*dy) || 0.001;
      joints[i] = {
        x: joints[i+1].x + (dx/d)*lengths[i],
        y: joints[i+1].y + (dy/d)*lengths[i],
      };
    }
    joints[0] = {...base};
    for (let i = 1; i < n; i++) {
      const dx = joints[i].x - joints[i-1].x;
      const dy = joints[i].y - joints[i-1].y;
      const d = Math.sqrt(dx*dx + dy*dy) || 0.001;
      joints[i] = {
        x: joints[i-1].x + (dx/d)*lengths[i-1],
        y: joints[i-1].y + (dy/d)*lengths[i-1],
      };
    }
    const err = Math.sqrt((joints[n-1].x-target.x)**2 + (joints[n-1].y-target.y)**2);
    if (err < tolerance) {
      console.log("Converged in " + (iter+1) + " iterations, error: " + err.toFixed(5));
      break;
    }
  }
  return joints;
}

const chain = [{x:0,y:0}, {x:0,y:0.5}, {x:0,y:1.0}, {x:0,y:1.4}, {x:0,y:1.7}];
const result = fabrik(chain, {x:1.0, y:0.8}, 0.001, 50);

const a1 = Math.atan2(result[1].y-result[0].y, result[1].x-result[0].x);
const a2 = Math.atan2(result[2].y-result[1].y, result[2].x-result[1].x) - a1;
robot.setJoint(1, a1);
robot.setJoint(2, a2);

console.log("End effector: (" + result[4].x.toFixed(3) + ", " + result[4].y.toFixed(3) + ")");
`,
    },
    {
      id: 'rrt-path-planner', name: 'RRT Path Planner',
      description: 'Rapidly-exploring Random Tree for collision-free motion planning',
      category: 'Control', categoryId: 'control', language: 'typescript',
      tags: ['rrt', 'planning', 'collision-free'],
      code: `// RRT — Rapidly-exploring Random Tree path planner
// Finds collision-free paths in joint space

function rrt(start, goal, obstacles, maxIter, stepSize) {
  const tree = [{ pos: start, parent: -1 }];
  
  function collides(p) {
    for (const obs of obstacles) {
      const dx = p[0]-obs.x, dy = p[1]-obs.y;
      if (Math.sqrt(dx*dx+dy*dy) < obs.r) return true;
    }
    return false;
  }
  
  function nearest(point) {
    let minD = Infinity, idx = 0;
    for (let i = 0; i < tree.length; i++) {
      const dx = tree[i].pos[0]-point[0], dy = tree[i].pos[1]-point[1];
      const d = Math.sqrt(dx*dx+dy*dy);
      if (d < minD) { minD = d; idx = i; }
    }
    return idx;
  }
  
  function steer(from, to) {
    const dx = to[0]-from[0], dy = to[1]-from[1];
    const d = Math.sqrt(dx*dx+dy*dy);
    if (d <= stepSize) return to;
    return [from[0]+dx/d*stepSize, from[1]+dy/d*stepSize];
  }
  
  for (let i = 0; i < maxIter; i++) {
    const sample = Math.random() < 0.1 ? goal :
      [Math.random()*Math.PI*2-Math.PI, Math.random()*Math.PI*2-Math.PI];
    
    const nIdx = nearest(sample);
    const newPos = steer(tree[nIdx].pos, sample);
    
    if (!collides(newPos)) {
      tree.push({ pos: newPos, parent: nIdx });
      
      const dx = newPos[0]-goal[0], dy = newPos[1]-goal[1];
      if (Math.sqrt(dx*dx+dy*dy) < stepSize) {
        // Trace path
        const path = [];
        let idx = tree.length-1;
        while (idx >= 0) { path.unshift(tree[idx].pos); idx = tree[idx].parent; }
        return path;
      }
    }
  }
  return null;
}

const obstacles = [{x:0.5,y:0.5,r:0.3}, {x:-0.8,y:0.2,r:0.25}, {x:0.2,y:-0.6,r:0.2}];
const path = rrt([0,0], [1.0,0.8], obstacles, 2000, 0.15);

if (path) {
  console.log("Path found! " + path.length + " waypoints");
  const last = path[path.length-1];
  robot.setJoint(1, last[0]);
  robot.setJoint(2, last[1]);
  path.forEach((p, i) => {
    if (i % Math.ceil(path.length/5) === 0)
      console.log("  wp" + i + ": (" + p[0].toFixed(2) + ", " + p[1].toFixed(2) + ")");
  });
} else {
  console.log("No path found in 2000 iterations");
}
`,
    },
    {
      id: 'spline-interpolation', name: 'Spline Interpolation',
      description: 'Catmull-Rom spline for smooth multi-waypoint trajectories',
      category: 'Control', categoryId: 'control', language: 'typescript',
      tags: ['spline', 'interpolation', 'smooth'],
      code: `// Catmull-Rom Spline — smooth interpolation through waypoints
// Used for multi-point trajectory generation

function catmullRom(p0, p1, p2, p3, t) {
  const t2 = t*t, t3 = t2*t;
  return 0.5 * (
    (2*p1) +
    (-p0 + p2) * t +
    (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
    (-p0 + 3*p1 - 3*p2 + p3) * t3
  );
}

function interpolateSpline(waypoints, samplesPerSegment) {
  const result = [];
  const n = waypoints.length;
  
  for (let i = 0; i < n - 1; i++) {
    const p0 = waypoints[Math.max(0, i-1)];
    const p1 = waypoints[i];
    const p2 = waypoints[Math.min(n-1, i+1)];
    const p3 = waypoints[Math.min(n-1, i+2)];
    
    for (let s = 0; s < samplesPerSegment; s++) {
      const t = s / samplesPerSegment;
      result.push({
        j1: catmullRom(p0.j1, p1.j1, p2.j1, p3.j1, t),
        j2: catmullRom(p0.j2, p1.j2, p2.j2, p3.j2, t),
      });
    }
  }
  result.push(waypoints[n-1]);
  return result;
}

const waypoints = [
  { j1: 0, j2: 0 },
  { j1: Math.PI/6, j2: -Math.PI/4 },
  { j1: Math.PI/3, j2: -Math.PI/6 },
  { j1: Math.PI/4, j2: -Math.PI/3 },
  { j1: 0, j2: 0 },
];

const smooth = interpolateSpline(waypoints, 30);

console.log("Waypoints: " + waypoints.length);
console.log("Smooth points: " + smooth.length);

for (const pt of smooth) {
  robot.setJoint(1, pt.j1);
  robot.setJoint(2, pt.j2);
}

console.log("Spline interpolation complete");
console.log("Final: j1=" + (smooth[smooth.length-1].j1*180/Math.PI).toFixed(1) + "° j2=" + (smooth[smooth.length-1].j2*180/Math.PI).toFixed(1) + "°");
`,
    },
  ],
  perception: [
    {
      id: 'lidar-scan', name: 'LiDAR Scanner',
      description: 'Planar LiDAR with ray casting, noise, and point cloud',
      category: 'Perception', categoryId: 'perception', language: 'typescript',
      tags: ['lidar', 'sensor', 'point-cloud'],
      code: `// LiDAR Scanner — 2D planar ray casting with gaussian noise

function gaussNoise() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v);
}

function rayCircle(ox, oy, dx, dy, cx, cy, r) {
  const fx = ox-cx, fy = oy-cy;
  const a = dx*dx + dy*dy;
  const b = 2*(fx*dx + fy*dy);
  const c = fx*fx + fy*fy - r*r;
  const disc = b*b - 4*a*c;
  if (disc < 0) return -1;
  const t = (-b - Math.sqrt(disc)) / (2*a);
  return t >= 0 ? t : -1;
}

const obstacles = [
  {x: 2, y: 1, r: 0.3},
  {x: -1.5, y: -2, r: 0.4},
  {x: 1, y: -1.5, r: 0.35},
  {x: 3, y: 0, r: 0.5},
];

const numRays = 64;
const fov = Math.PI;
const maxRange = 5.0;
const noiseStd = 0.02;
const rays = [];

for (let i = 0; i < numRays; i++) {
  const angle = -fov/2 + (i/(numRays-1)) * fov;
  const dx = Math.cos(angle), dy = Math.sin(angle);
  let minDist = maxRange;
  let hit = false;

  for (const obs of obstacles) {
    const t = rayCircle(0, 0, dx, dy, obs.x, obs.y, obs.r);
    if (t > 0 && t < minDist) { minDist = t; hit = true; }
  }

  if (hit) minDist += gaussNoise() * noiseStd;
  rays.push({ angle, distance: Math.max(0, minDist), hit });
}

robot.showLidar(rays);

const hits = rays.filter(r => r.hit);
console.log("Rays: " + numRays + " | Hits: " + hits.length);
console.log("Nearest: " + Math.min(...hits.map(r=>r.distance)).toFixed(2) + "m");
console.log("Farthest: " + Math.max(...hits.map(r=>r.distance)).toFixed(2) + "m");
`,
    },
    {
      id: 'depth-camera', name: 'Depth Camera',
      description: 'Pinhole depth camera with ray-sphere intersection',
      category: 'Perception', categoryId: 'perception', language: 'typescript',
      tags: ['depth', 'camera', 'rgbd'],
      code: `// Depth Camera — pinhole model with ray-sphere intersection

const width = 32, height = 24;
const fov = Math.PI / 3;
const focalLength = (width / 2) / Math.tan(fov / 2);

const spheres = [
  { x: 0, y: 0, z: 3, r: 0.5 },
  { x: 1, y: 0.5, z: 4, r: 0.3 },
  { x: -0.8, y: -0.3, z: 2.5, r: 0.4 },
];

let minDepth = Infinity, maxDepth = 0, pixelsHit = 0;

for (let py = 0; py < height; py++) {
  for (let px = 0; px < width; px++) {
    const rx = (px - width/2) / focalLength;
    const ry = (py - height/2) / focalLength;
    const rz = 1.0;
    const len = Math.sqrt(rx*rx + ry*ry + rz*rz);
    const dx = rx/len, dy = ry/len, dz = rz/len;

    for (const s of spheres) {
      const a = 1;
      const b = 2*(dx*(-s.x) + dy*(-s.y) + dz*(-s.z));
      const c = s.x*s.x + s.y*s.y + s.z*s.z - s.r*s.r;
      const disc = b*b - 4*a*c;
      if (disc >= 0) {
        const t = (-b - Math.sqrt(disc)) / 2;
        if (t > 0.1) {
          pixelsHit++;
          minDepth = Math.min(minDepth, t);
          maxDepth = Math.max(maxDepth, t);
        }
      }
    }
  }
}

console.log("Resolution: " + width + "x" + height);
console.log("Pixels with depth: " + pixelsHit + "/" + (width*height));
console.log("Depth range: " + minDepth.toFixed(2) + "m - " + maxDepth.toFixed(2) + "m");
console.log("FOV: " + (fov * 180 / Math.PI).toFixed(0) + "°");
`,
    },
    {
      id: 'imu-sensor', name: 'IMU Sensor',
      description: 'Complementary filter fusing accelerometer and gyroscope',
      category: 'Perception', categoryId: 'perception', language: 'typescript',
      tags: ['imu', 'accelerometer', 'gyroscope'],
      code: `// IMU Sensor — complementary filter for orientation estimation

const gravity = 9.81;
const dt = 0.01;
const alpha = 0.98;

let pitch = 0, roll = 0;

for (let i = 0; i < 200; i++) {
  const t = i * dt;
  const truePitch = Math.sin(t * 2) * 0.3;
  const trueRoll = Math.cos(t * 1.5) * 0.2;

  const ax = -gravity * Math.sin(truePitch) + (Math.random()-0.5)*0.1;
  const ay = gravity * Math.sin(trueRoll) + (Math.random()-0.5)*0.1;
  const az = gravity * Math.cos(truePitch) * Math.cos(trueRoll) + (Math.random()-0.5)*0.1;

  const gx = Math.cos(t*2)*0.6 + 0.01 + (Math.random()-0.5)*0.02;
  const gy = -Math.sin(t*1.5)*0.3 + 0.005 + (Math.random()-0.5)*0.02;

  const accPitch = Math.atan2(ax, Math.sqrt(ay*ay + az*az));
  const accRoll = Math.atan2(ay, az);

  pitch = alpha * (pitch + gx * dt) + (1-alpha) * accPitch;
  roll = alpha * (roll + gy * dt) + (1-alpha) * accRoll;

  if (i % 50 === 0) {
    console.log("t=" + t.toFixed(2) + "s | Pitch: " + (pitch*180/Math.PI).toFixed(1) + "° (true: " + (truePitch*180/Math.PI).toFixed(1) + "°)");
  }
}

robot.setJoint(1, pitch);
robot.setJoint(2, roll);
console.log("Filter coefficient: " + alpha);
`,
    },
    {
      id: 'object-detection', name: 'Object Detector (NMS)',
      description: 'Non-maximum suppression for bounding box detection',
      category: 'Perception', categoryId: 'perception', language: 'typescript',
      tags: ['detection', 'nms', 'vision'],
      code: `// Object Detection — Non-Maximum Suppression (NMS)
// Core post-processing in YOLO, SSD, Faster R-CNN

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x+a.w, b.x+b.w), y2 = Math.min(a.y+a.h, b.y+b.h);
  const inter = Math.max(0, x2-x1) * Math.max(0, y2-y1);
  const union = a.w*a.h + b.w*b.h - inter;
  return union > 0 ? inter/union : 0;
}

function nms(boxes, iouThreshold) {
  const sorted = [...boxes].sort((a,b) => b.score - a.score);
  const keep = [];
  const suppressed = new Set();
  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;
    keep.push(sorted[i]);
    for (let j = i+1; j < sorted.length; j++) {
      if (iou(sorted[i], sorted[j]) > iouThreshold) suppressed.add(j);
    }
  }
  return keep;
}

const rawDetections = [
  { x:100, y:50, w:80, h:120, score:0.92, label:"robot" },
  { x:105, y:55, w:75, h:115, score:0.88, label:"robot" },
  { x:300, y:100, w:60, h:60, score:0.75, label:"sensor" },
  { x:310, y:105, w:55, h:55, score:0.71, label:"sensor" },
  { x:200, y:200, w:50, h:80, score:0.65, label:"gripper" },
  { x:102, y:52, w:78, h:118, score:0.85, label:"robot" },
];

console.log("Raw detections: " + rawDetections.length);
const filtered = nms(rawDetections, 0.5);
console.log("After NMS: " + filtered.length);
filtered.forEach(d => {
  console.log("  " + d.label + " — score: " + d.score.toFixed(2) + " @ [" + d.x + "," + d.y + "," + d.w + "," + d.h + "]");
});
`,
    },
    {
      id: 'kalman-filter', name: 'Kalman Filter',
      description: '1D Kalman filter for noisy sensor fusion',
      category: 'Perception', categoryId: 'perception', language: 'typescript',
      tags: ['kalman', 'filter', 'estimation'],
      code: `// 1D Kalman Filter — optimal state estimation from noisy sensors
// Tracks position of a moving object

let x = 0;       // Estimated state
let P = 1;       // Estimate covariance
const Q = 0.01;  // Process noise
const R = 0.5;   // Measurement noise
const dt = 0.1;
const velocity = 0.5; // True velocity

const measurements = [];
const estimates = [];

for (let i = 0; i < 50; i++) {
  const truePos = velocity * i * dt;
  
  // Predict step
  x = x + velocity * dt;  // State prediction
  P = P + Q;              // Covariance prediction
  
  // Measurement (noisy)
  const z = truePos + (Math.random() - 0.5) * 2 * Math.sqrt(R);
  
  // Update step
  const K = P / (P + R);  // Kalman gain
  x = x + K * (z - x);    // State update
  P = (1 - K) * P;        // Covariance update
  
  measurements.push(z);
  estimates.push(x);
  
  if (i % 10 === 0) {
    console.log("t=" + (i*dt).toFixed(1) + "s | True: " + truePos.toFixed(2) + 
      " | Measured: " + z.toFixed(2) + " | Kalman: " + x.toFixed(2) + 
      " | Gain: " + K.toFixed(3));
  }
}

const trueF = velocity * 49 * dt;
const errMeas = Math.abs(measurements[49] - trueF);
const errKalman = Math.abs(estimates[49] - trueF);
console.log("Final measurement error: " + errMeas.toFixed(3));
console.log("Final Kalman error: " + errKalman.toFixed(3));
console.log("Improvement: " + ((1 - errKalman/errMeas)*100).toFixed(0) + "%");

robot.setJoint(1, x * 0.3);
`,
    },
  ],
  hardware: [
    {
      id: 'servo-control', name: 'Servo Motor',
      description: 'Velocity-limited servo with thermal model',
      category: 'Hardware', categoryId: 'hardware', language: 'typescript',
      tags: ['servo', 'motor', 'actuator'],
      code: `// Servo Motor Simulation — velocity-limited with thermal model

class Servo {
  constructor(maxVel, maxAngle, damping) {
    this.angle = 0; this.target = 0;
    this.velocity = 0; this.temp = 25;
    this.maxVel = maxVel; this.maxAngle = maxAngle;
    this.damping = damping;
  }
  setTarget(a) { this.target = Math.max(-this.maxAngle, Math.min(this.maxAngle, a)); }
  step(dt) {
    const err = this.target - this.angle;
    const desired = err * 10;
    this.velocity = Math.max(-this.maxVel, Math.min(this.maxVel, desired)) * this.damping;
    this.angle += this.velocity * dt;
    this.angle = Math.max(-this.maxAngle, Math.min(this.maxAngle, this.angle));
    this.temp += Math.abs(this.velocity) * 0.01 * dt;
    this.temp -= (this.temp - 25) * 0.005 * dt;
    return this.angle;
  }
}

const base = new Servo(5.24, Math.PI, 0.92);
const shoulder = new Servo(4.82, Math.PI, 0.90);
const elbow = new Servo(4.82, Math.PI, 0.90);

base.setTarget(0);
shoulder.setTarget(Math.PI / 4);
elbow.setTarget(-Math.PI / 3);

const dt = 0.016;
for (let i = 0; i < 180; i++) {
  robot.setJoint(0, base.step(dt));
  robot.setJoint(1, shoulder.step(dt));
  robot.setJoint(2, elbow.step(dt));
}

console.log("Base: " + (base.angle*180/Math.PI).toFixed(1) + "° | " + base.temp.toFixed(1) + "°C");
console.log("Shoulder: " + (shoulder.angle*180/Math.PI).toFixed(1) + "° | " + shoulder.temp.toFixed(1) + "°C");
console.log("Elbow: " + (elbow.angle*180/Math.PI).toFixed(1) + "° | " + elbow.temp.toFixed(1) + "°C");
`,
    },
    {
      id: 'gripper-control', name: 'Parallel Gripper',
      description: 'Force-controlled gripper with object detection',
      category: 'Hardware', categoryId: 'hardware', language: 'typescript',
      tags: ['gripper', 'grasp', 'end-effector'],
      code: `// Parallel Gripper — force-controlled with object detection

let width = 0.08;
let target = 0.08;
let force = 0;
let state = "open";
const objectWidth = 0.03;
const speed = 0.05;
const maxForce = 40;

target = 0;
state = "closing";

const dt = 0.016;
for (let i = 0; i < 120; i++) {
  const err = target - width;
  const delta = Math.max(-speed*dt, Math.min(speed*dt, err));
  const newWidth = width + delta;

  if (newWidth <= objectWidth && state === "closing") {
    width = objectWidth;
    force = Math.min(maxForce, (objectWidth - target) * 500);
    state = "gripping";
    console.log("Object detected at " + (width*1000).toFixed(1) + "mm");
    console.log("Grip force: " + force.toFixed(1) + "N");
    break;
  }

  width = Math.max(0, Math.min(0.08, newWidth));
  robot.setGripper(width / 0.08);
}

console.log("Holding object...");

target = 0.08;
state = "opening";
force = 0;

for (let i = 0; i < 80; i++) {
  width += speed * dt;
  width = Math.min(0.08, width);
  robot.setGripper(width / 0.08);
}

console.log("Released. Width: " + (width*1000).toFixed(1) + "mm");
`,
    },
    {
      id: 'stepper-motor', name: 'Stepper Motor',
      description: 'Microstepping driver with trapezoidal velocity profile',
      category: 'Hardware', categoryId: 'hardware', language: 'typescript',
      tags: ['stepper', 'motor', 'microstepping'],
      code: `// Stepper Motor — trapezoidal velocity profile with microstepping

const stepsPerRev = 200;
const microsteps = 16;
const totalSteps = stepsPerRev * microsteps;
const maxSpeed = 1000;
const accel = 5000;
const targetSteps = 800;

let position = 0;
let speed = 0;
let step = 0;
const dt = 0.001;

const accelSteps = Math.ceil((maxSpeed * maxSpeed) / (2 * accel));
const decelStart = targetSteps - accelSteps;

const profile = [];
while (position < targetSteps) {
  if (position < accelSteps) speed = Math.min(speed + accel * dt, maxSpeed);
  else if (position >= decelStart) speed = Math.max(speed - accel * dt, 50);
  position += speed * dt;
  step++;
  if (step % 200 === 0) profile.push({ pos: position, speed: speed.toFixed(0) });
}

const finalAngle = (position / totalSteps) * 2 * Math.PI;
robot.setJoint(1, finalAngle);

console.log("Steps/rev: " + totalSteps + " (" + microsteps + "x microstepping)");
console.log("Target: " + targetSteps + " steps (" + (targetSteps*360/totalSteps).toFixed(1) + "°)");
console.log("Peak speed: " + maxSpeed + " sps (" + (maxSpeed*60/totalSteps).toFixed(1) + " RPM)");
profile.forEach(p => console.log("  pos=" + Math.round(p.pos) + " speed=" + p.speed + " sps"));
`,
    },
    {
      id: 'encoder-feedback', name: 'Rotary Encoder',
      description: 'Quadrature encoder with velocity estimation',
      category: 'Hardware', categoryId: 'hardware', language: 'typescript',
      tags: ['encoder', 'feedback', 'quadrature'],
      code: `// Quadrature Encoder — position and velocity estimation

const ppr = 1024;
const dt = 0.001;

class QuadratureEncoder {
  constructor(ppr) {
    this.ppr = ppr; this.count = 0; this.prevCount = 0;
    this.velocity = 0; this.position = 0;
  }
  update(trueAngle) {
    this.count = Math.round(trueAngle / (2 * Math.PI) * this.ppr * 4);
    this.position = (this.count / (this.ppr * 4)) * 2 * Math.PI;
    this.velocity = (this.count - this.prevCount) / (this.ppr * 4) * 2 * Math.PI / dt;
    this.prevCount = this.count;
  }
}

const encoder = new QuadratureEncoder(ppr);
let trueAngle = 0;
let trueVelocity = 0;
const results = [];

for (let i = 0; i < 2000; i++) {
  const t = i * dt;
  if (t < 1.0) trueVelocity = t * 10;
  else trueVelocity = Math.max(0, 10 - (t-1.0) * 10);
  trueAngle += trueVelocity * dt;
  encoder.update(trueAngle);
  if (i % 400 === 0) results.push({ t: t.toFixed(2), pos: encoder.position.toFixed(3), vel: encoder.velocity.toFixed(1) });
  robot.setJoint(1, encoder.position % (Math.PI * 2));
}

console.log("Encoder: " + ppr + " PPR, 4x = " + (ppr*4) + " counts/rev");
console.log("Resolution: " + (360/(ppr*4)).toFixed(4) + "°/count");
results.forEach(r => console.log("t=" + r.t + "s pos=" + r.pos + "rad vel=" + r.vel + "rad/s"));
`,
    },
    {
      id: 'dc-motor', name: 'DC Motor Model',
      description: 'Brushed DC motor with back-EMF and torque curves',
      category: 'Hardware', categoryId: 'hardware', language: 'typescript',
      tags: ['dc-motor', 'back-emf', 'torque'],
      code: `// DC Motor Model — electrical + mechanical dynamics
// Includes back-EMF, armature resistance, and inertia

const R = 2.5;      // Resistance (Ω)
const L = 0.001;    // Inductance (H)
const Kt = 0.05;    // Torque constant (Nm/A)
const Ke = 0.05;    // Back-EMF constant (V·s/rad)
const J = 0.001;    // Inertia (kg·m²)
const B = 0.0005;   // Friction (Nm·s/rad)
const Vin = 12;     // Supply voltage (V)

let current = 0;    // Armature current (A)
let omega = 0;      // Angular velocity (rad/s)
let theta = 0;      // Position (rad)
const dt = 0.0001;  // 10kHz simulation

const snapshots = [];

for (let i = 0; i < 50000; i++) {
  const backEmf = Ke * omega;
  const dI = (Vin - R * current - backEmf) / L;
  current += dI * dt;
  
  const torque = Kt * current;
  const friction = B * omega;
  const dOmega = (torque - friction) / J;
  omega += dOmega * dt;
  theta += omega * dt;
  
  if (i % 10000 === 0) {
    snapshots.push({
      t: (i * dt * 1000).toFixed(0),
      rpm: (omega * 60 / (2*Math.PI)).toFixed(0),
      current: current.toFixed(2),
      torque: torque.toFixed(4),
    });
  }
}

console.log("DC Motor Simulation (12V supply)");
console.log("─────────────────────────────");
snapshots.forEach(s => {
  console.log("t=" + s.t + "ms | " + s.rpm + " RPM | I=" + s.current + "A | τ=" + s.torque + " Nm");
});
console.log("Steady-state: " + (omega * 60 / (2*Math.PI)).toFixed(0) + " RPM");

robot.setJoint(1, theta % (2 * Math.PI));
`,
    },
  ],
  simulation: [
    {
      id: 'robot-arm-sim', name: 'Robot Arm Simulation',
      description: 'Full 3-DOF arm with PID-controlled servos',
      category: 'Simulation', categoryId: 'simulation', language: 'typescript',
      tags: ['arm', 'simulation', 'full-stack'],
      code: `// Robot Arm — full simulation with PID + servo + IK

function solve2LinkIK(tx, ty, l1, l2) {
  const d = Math.sqrt(tx*tx + ty*ty);
  const clamped = Math.min(l1+l2-0.01, d);
  const s = d > l1+l2 ? clamped/d : 1;
  const x = tx*s, y = ty*s;
  const dSq = x*x + y*y;
  const cosA2 = (dSq - l1*l1 - l2*l2) / (2*l1*l2);
  const a2 = -Math.acos(Math.max(-1, Math.min(1, cosA2)));
  const a1 = Math.atan2(y, x) - Math.atan2(l2*Math.sin(a2), l1+l2*Math.cos(a2));
  return [a1, a2];
}

const target = { x: 1.0, y: 0.6 };
const [a1, a2] = solve2LinkIK(target.x, target.y, 1.0, 0.8);

console.log("Target: (" + target.x + ", " + target.y + ")");
console.log("IK solution: θ1=" + (a1*180/Math.PI).toFixed(1) + "° θ2=" + (a2*180/Math.PI).toFixed(1) + "°");

let s1 = 0, s2 = 0;
for (let i = 0; i < 150; i++) {
  s1 += (a1 - s1) * 0.08;
  s2 += (a2 - s2) * 0.08;
  robot.setJoint(1, s1);
  robot.setJoint(2, s2);
}

for (let i = 0; i < 60; i++) robot.setGripper(1 - i/60);

const ex = Math.cos(s1) + 0.8*Math.cos(s1+s2);
const ey = Math.sin(s1) + 0.8*Math.sin(s1+s2);
console.log("Reached: (" + ex.toFixed(3) + ", " + ey.toFixed(3) + ")");
console.log("Error: " + Math.sqrt((ex-target.x)**2+(ey-target.y)**2).toFixed(4) + "m");
`,
    },
    {
      id: 'pick-and-place', name: 'Pick & Place Task',
      description: 'Complete pick-and-place manipulation sequence',
      category: 'Simulation', categoryId: 'simulation', language: 'typescript',
      tags: ['pick-place', 'task', 'sequence'],
      code: `// Pick & Place — multi-step manipulation sequence

function ik(tx, ty) {
  const d = Math.min(1.79, Math.sqrt(tx*tx + ty*ty));
  const s = d / Math.sqrt(tx*tx + ty*ty);
  const x = tx*s, y = ty*s;
  const cosA2 = (x*x+y*y - 1 - 0.64) / 1.6;
  const a2 = -Math.acos(Math.max(-1, Math.min(1, cosA2)));
  const a1 = Math.atan2(y,x) - Math.atan2(0.8*Math.sin(a2), 1+0.8*Math.cos(a2));
  return [a1, a2];
}

let j1 = 0, j2 = 0, grip = 1;

function animateTo(t1, t2, g, steps) {
  for (let i = 0; i < steps; i++) {
    j1 += (t1-j1)*0.1; j2 += (t2-j2)*0.1; grip += (g-grip)*0.1;
    robot.setJoint(1, j1); robot.setJoint(2, j2); robot.setGripper(grip);
  }
}

console.log(">> Approach pick position");
const [p1,p2] = ik(1.2, 0.3);
animateTo(p1, p2, 1, 80);

console.log(">> Closing gripper");
animateTo(p1, p2, 0.1, 40);

console.log(">> Lifting");
const [l1,l2] = ik(1.0, 1.0);
animateTo(l1, l2, 0.1, 60);

console.log(">> Moving to place");
const [d1,d2] = ik(0.5, 0.8);
animateTo(d1, d2, 0.1, 60);

console.log(">> Releasing");
animateTo(d1, d2, 1, 40);

console.log(">> Returning home");
animateTo(Math.PI/4, -Math.PI/4, 1, 80);
console.log("Pick and place complete ✓");
`,
    },
    {
      id: 'collision-detection', name: 'Collision Detection',
      description: 'Sphere-sphere and AABB collision for robot safety',
      category: 'Simulation', categoryId: 'simulation', language: 'typescript',
      tags: ['collision', 'safety', 'physics'],
      code: `// Collision Detection — sphere and AABB methods

function sphereCollision(a, b) {
  const dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
  const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
  const minDist = a.r + b.r;
  return {
    colliding: dist < minDist,
    distance: dist,
    penetration: Math.max(0, minDist - dist),
    normal: dist > 0 ? { x: dx/dist, y: dy/dist, z: dz/dist } : { x: 0, y: 1, z: 0 },
  };
}

const joints = [
  { x: 0, y: 0.3, z: 0, r: 0.07, name: "base" },
  { x: 0.5, y: 0.8, z: 0, r: 0.06, name: "shoulder" },
  { x: 1.2, y: 0.6, z: 0, r: 0.06, name: "elbow" },
  { x: 1.6, y: 0.4, z: 0, r: 0.04, name: "wrist" },
];

const obstacles = [
  { x: 1.0, y: 0.5, z: 0, r: 0.3, name: "box" },
  { x: -0.5, y: 0.8, z: 0.3, r: 0.2, name: "cylinder" },
];

let collisions = 0;
for (const joint of joints) {
  for (const obs of obstacles) {
    const result = sphereCollision(joint, obs);
    if (result.colliding) {
      collisions++;
      console.log("COLLISION: " + joint.name + " ↔ " + obs.name + " (penetration: " + result.penetration.toFixed(3) + "m)");
    }
  }
}

console.log("Joints: " + joints.length + " | Obstacles: " + obstacles.length);
console.log("Collisions found: " + collisions);
if (collisions === 0) console.log("Path is clear ✓");
`,
    },
    {
      id: 'spring-damper', name: 'Spring-Damper System',
      description: 'Mass-spring-damper physics simulation with energy tracking',
      category: 'Simulation', categoryId: 'simulation', language: 'typescript',
      tags: ['physics', 'spring', 'damper'],
      code: `// Spring-Damper System — vibration simulation with energy tracking
// Models compliant joints and shock absorption

const m = 1.0;    // Mass (kg)
const k = 50.0;   // Spring stiffness (N/m)
const c = 2.0;    // Damping coefficient (Ns/m)
const x0 = 0.5;   // Initial displacement (m)

let x = x0;       // Position
let v = 0;        // Velocity
const dt = 0.001;
const naturalFreq = Math.sqrt(k/m);
const dampingRatio = c / (2 * Math.sqrt(k * m));

console.log("Natural frequency: " + (naturalFreq/(2*Math.PI)).toFixed(2) + " Hz");
console.log("Damping ratio: " + dampingRatio.toFixed(3) + " (" + 
  (dampingRatio < 1 ? "underdamped" : dampingRatio === 1 ? "critically damped" : "overdamped") + ")");

let peakCount = 0;
let prevX = x;
const peaks = [];

for (let i = 0; i < 5000; i++) {
  const springForce = -k * x;
  const dampForce = -c * v;
  const a = (springForce + dampForce) / m;
  v += a * dt;
  x += v * dt;
  
  // Detect peaks
  if (i > 1 && prevX > x && prevX > 0.01) {
    peakCount++;
    peaks.push({ t: (i*dt).toFixed(3), amp: prevX.toFixed(4) });
  }
  prevX = x;
  
  robot.setJoint(1, x * 2); // Visualize displacement as joint angle
}

console.log("─────────────────────────────");
console.log("Oscillation peaks:");
peaks.slice(0, 6).forEach(p => console.log("  t=" + p.t + "s  amp=" + p.amp + "m"));
console.log("Total oscillations: " + peakCount);
console.log("Final displacement: " + (x*1000).toFixed(3) + "mm (settled: " + (Math.abs(x) < 0.001 ? "yes" : "no") + ")");
`,
    },
  ],
  ai: [
    {
      id: 'reinforcement-learning', name: 'RL Training Loop',
      description: 'REINFORCE policy gradient for reaching tasks',
      category: 'AI / Agents', categoryId: 'ai', language: 'typescript',
      tags: ['rl', 'training', 'policy'],
      code: `// REINFORCE — policy gradient for robot reaching

class Policy {
  constructor(inDim, outDim) {
    this.w = [];
    for (let i = 0; i < outDim; i++)
      this.w.push(Array.from({length: inDim}, () => (Math.random()-0.5)*0.1));
    this.lr = 0.01;
  }
  forward(state) {
    return this.w.map(wi => Math.tanh(wi.reduce((s,w,j) => s + w*(state[j]||0), 0)));
  }
  update(state, action, reward) {
    for (let i = 0; i < this.w.length; i++)
      for (let j = 0; j < this.w[i].length; j++)
        this.w[i][j] += this.lr * reward * action[i] * (state[j]||0);
  }
}

const policy = new Policy(5, 3);
let bestReward = -Infinity;

for (let ep = 0; ep < 80; ep++) {
  const tx = 0.6 + Math.random()*0.8;
  const ty = 0.2 + Math.random()*0.6;
  const angles = [0, Math.PI/4, -Math.PI/4];
  let totalReward = 0;

  for (let step = 0; step < 30; step++) {
    const state = [...angles, tx, ty];
    const action = policy.forward(state);
    for (let i = 0; i < 3; i++) angles[i] += action[i] * 0.05;

    const ex = Math.cos(angles[1]) + 0.8*Math.cos(angles[1]+angles[2]);
    const ey = Math.sin(angles[1]) + 0.8*Math.sin(angles[1]+angles[2]);
    const dist = Math.sqrt((ex-tx)**2 + (ey-ty)**2);
    const reward = -dist;
    totalReward += reward;
    policy.update(state, action, reward);
    robot.setJoint(0, angles[0]); robot.setJoint(1, angles[1]); robot.setJoint(2, angles[2]);
  }

  if (totalReward > bestReward) bestReward = totalReward;
  if (ep % 20 === 0) console.log("Episode " + ep + " | Reward: " + totalReward.toFixed(1));
}

console.log("Best reward: " + bestReward.toFixed(1));
`,
    },
    {
      id: 'behavior-tree', name: 'Behavior Tree',
      description: 'Hierarchical BT for autonomous decision-making',
      category: 'AI / Agents', categoryId: 'ai', language: 'typescript',
      tags: ['behavior-tree', 'decision', 'autonomy'],
      code: `// Behavior Tree — hierarchical decision making

const SUCCESS = "success", FAILURE = "failure", RUNNING = "running";

function Sequence(...children) {
  return (ctx) => { for (const c of children) { const r = c(ctx); if (r !== SUCCESS) return r; } return SUCCESS; };
}
function Selector(...children) {
  return (ctx) => { for (const c of children) { const r = c(ctx); if (r !== FAILURE) return r; } return FAILURE; };
}
function Action(name, fn) {
  return (ctx) => { console.log("[BT] " + name); return fn(ctx); };
}

const ctx = { battery: 75, objectDetected: true, objectDist: 0.5, holding: false };

const tree = Selector(
  Sequence(
    Action("CheckBattery", c => c.battery > 20 ? SUCCESS : FAILURE),
    Selector(
      Sequence(
        Action("DetectObject", c => c.objectDetected ? SUCCESS : FAILURE),
        Action("Approach", c => { c.objectDist -= 0.1; return c.objectDist <= 0.1 ? SUCCESS : RUNNING; }),
        Action("Grasp", c => { c.holding = true; robot.setGripper(0.2); return SUCCESS; }),
        Action("Deliver", c => { robot.setJoint(1, Math.PI/3); return SUCCESS; }),
        Action("Release", c => { c.holding = false; robot.setGripper(1); return SUCCESS; }),
      ),
      Action("Patrol", c => { robot.setJoint(0, Math.random()*0.5); return SUCCESS; }),
    ),
  ),
  Action("ReturnToCharger", c => { console.log("Low battery!"); return SUCCESS; }),
);

for (let i = 0; i < 5; i++) {
  console.log("--- Tick " + i + " ---");
  const result = tree(ctx);
  console.log("Result: " + result);
}
`,
    },
    {
      id: 'state-machine', name: 'Finite State Machine',
      description: 'Event-driven FSM for operational modes',
      category: 'AI / Agents', categoryId: 'ai', language: 'typescript',
      tags: ['fsm', 'state-machine', 'control'],
      code: `// Finite State Machine — robot operational modes

class FSM {
  constructor() { this.states = {}; this.current = null; this.transitions = []; }
  addState(name, onEnter, onUpdate, onExit) { this.states[name] = { onEnter, onUpdate, onExit }; }
  addTransition(from, to, condition) { this.transitions.push({ from, to, condition }); }
  start(name) { this.current = name; this.states[name].onEnter(); }
  update(ctx) {
    for (const t of this.transitions) {
      if (t.from === this.current && t.condition(ctx)) {
        this.states[this.current].onExit();
        this.current = t.to;
        this.states[t.to].onEnter();
        break;
      }
    }
    this.states[this.current].onUpdate(ctx);
  }
}

const fsm = new FSM();
let angle = 0;

fsm.addState("IDLE",
  () => console.log(">> IDLE"),
  () => { robot.setJoint(1, 0); robot.setJoint(2, 0); },
  () => console.log("<< IDLE")
);
fsm.addState("SEARCHING",
  () => console.log(">> SEARCH"),
  () => { angle += 0.05; robot.setJoint(0, Math.sin(angle) * 0.5); },
  () => console.log("<< SEARCH")
);
fsm.addState("REACHING",
  () => console.log(">> REACH"),
  () => { robot.setJoint(1, Math.PI/4); robot.setJoint(2, -Math.PI/6); },
  () => console.log("<< REACH")
);
fsm.addState("GRASPING",
  () => { console.log(">> GRASP"); robot.setGripper(0.1); },
  () => {},
  () => console.log("<< GRASP")
);

fsm.addTransition("IDLE", "SEARCHING", c => c.tick > 2);
fsm.addTransition("SEARCHING", "REACHING", c => c.tick > 6);
fsm.addTransition("REACHING", "GRASPING", c => c.tick > 9);
fsm.addTransition("GRASPING", "IDLE", c => c.tick > 11);

fsm.start("IDLE");
for (let tick = 0; tick < 14; tick++) {
  fsm.update({ tick });
  console.log("Tick " + tick + " — State: " + fsm.current);
}
`,
    },
    {
      id: 'a-star-pathfinding', name: 'A* Pathfinding',
      description: 'Grid-based A* pathfinding with diagonal movement',
      category: 'AI / Agents', categoryId: 'ai', language: 'typescript',
      tags: ['astar', 'pathfinding', 'grid'],
      code: `// A* Pathfinding — grid-based with diagonal movement
// Used for mobile robot navigation planning

function aStar(grid, start, goal) {
  const rows = grid.length, cols = grid[0].length;
  const key = (r,c) => r + "," + c;
  
  const open = [{ r: start[0], c: start[1], g: 0, f: 0 }];
  const gScore = { [key(start[0],start[1])]: 0 };
  const parent = {};
  const closed = new Set();
  
  const dirs = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
  
  function heuristic(r, c) {
    return Math.sqrt((r-goal[0])**2 + (c-goal[1])**2);
  }
  
  while (open.length > 0) {
    open.sort((a,b) => a.f - b.f);
    const curr = open.shift();
    const ck = key(curr.r, curr.c);
    
    if (curr.r === goal[0] && curr.c === goal[1]) {
      const path = [];
      let k = ck;
      while (k) { const [r,c] = k.split(",").map(Number); path.unshift([r,c]); k = parent[k]; }
      return path;
    }
    
    closed.add(ck);
    
    for (const [dr, dc] of dirs) {
      const nr = curr.r+dr, nc = curr.c+dc;
      if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
      if (grid[nr][nc] === 1) continue;
      const nk = key(nr, nc);
      if (closed.has(nk)) continue;
      
      const cost = Math.abs(dr)+Math.abs(dc) === 2 ? 1.414 : 1;
      const ng = curr.g + cost;
      
      if (ng < (gScore[nk] ?? Infinity)) {
        gScore[nk] = ng;
        parent[nk] = ck;
        open.push({ r: nr, c: nc, g: ng, f: ng + heuristic(nr, nc) });
      }
    }
  }
  return null;
}

// 0 = free, 1 = obstacle
const grid = [
  [0,0,0,0,0,0,0,0,0,0],
  [0,0,0,1,1,1,0,0,0,0],
  [0,0,0,0,0,1,0,0,0,0],
  [0,1,1,0,0,0,0,1,0,0],
  [0,0,1,0,0,0,0,1,0,0],
  [0,0,0,0,1,0,0,0,0,0],
  [0,0,0,0,1,0,0,0,0,0],
  [0,0,0,0,0,0,1,1,0,0],
  [0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
];

const path = aStar(grid, [0,0], [9,9]);
if (path) {
  console.log("Path found! Length: " + path.length + " steps");
  console.log("Cost: " + (path.length * 0.5).toFixed(1) + " meters");
  path.forEach((p, i) => {
    if (i % 3 === 0) console.log("  [" + p[0] + "," + p[1] + "]");
  });
  robot.setJoint(1, Math.PI/4);
} else {
  console.log("No path found!");
}
`,
    },
  ],
  communication: [
    {
      id: 'pubsub-messaging', name: 'Pub/Sub Messaging',
      description: 'ROS-style topic-based publish/subscribe system',
      category: 'Communication', categoryId: 'communication', language: 'typescript',
      tags: ['ros', 'pubsub', 'messaging'],
      code: `// Pub/Sub Messaging — ROS-style topic system

class MessageBus {
  constructor() { this.topics = {}; }
  subscribe(topic, callback) {
    if (!this.topics[topic]) this.topics[topic] = [];
    this.topics[topic].push(callback);
    return () => { this.topics[topic] = this.topics[topic].filter(cb => cb !== callback); };
  }
  publish(topic, data) {
    const subs = this.topics[topic] || [];
    subs.forEach(cb => cb(data));
    return subs.length;
  }
  getTopics() { return Object.keys(this.topics); }
}

const bus = new MessageBus();
const logs = [];

bus.subscribe("/joint_states", (msg) => {
  logs.push("Joint update: " + msg.joints.map(j => j.toFixed(2)).join(", "));
});
bus.subscribe("/lidar/scan", (msg) => {
  logs.push("LiDAR: " + msg.ranges.length + " pts, min=" + Math.min(...msg.ranges).toFixed(2) + "m");
});
bus.subscribe("/cmd_vel", (msg) => {
  logs.push("Vel cmd: lin=" + msg.linear.toFixed(2) + " ang=" + msg.angular.toFixed(2));
  robot.setJoint(0, msg.angular);
});
bus.subscribe("/gripper/command", (msg) => {
  logs.push("Gripper: " + msg.action + " " + (msg.width*1000).toFixed(0) + "mm");
  robot.setGripper(msg.width / 0.08);
});

bus.publish("/joint_states", { joints: [0, Math.PI/4, -Math.PI/3], timestamp: Date.now() });
bus.publish("/lidar/scan", { ranges: Array.from({length: 64}, () => 0.5 + Math.random()*4) });
bus.publish("/cmd_vel", { linear: 0.5, angular: 0.3 });
bus.publish("/gripper/command", { action: "close", width: 0.03 });

robot.setJoint(1, Math.PI/3);
robot.setJoint(2, -Math.PI/4);

console.log("Active topics: " + bus.getTopics().join(", "));
console.log("Messages processed: " + logs.length);
logs.forEach(l => console.log("  " + l));
`,
    },
    {
      id: 'command-protocol', name: 'Command Protocol',
      description: 'Structured command-response protocol for serial comms',
      category: 'Communication', categoryId: 'communication', language: 'typescript',
      tags: ['protocol', 'command', 'serial'],
      code: `// Command Protocol — structured robot communication

class CommandProtocol {
  constructor() { this.seq = 0; this.handlers = {}; this.log = []; }
  register(cmd, handler) { this.handlers[cmd] = handler; }
  send(cmd, params) {
    const packet = { seq: this.seq++, cmd, params, timestamp: Date.now() };
    this.log.push({ ...packet, direction: "TX" });
    const handler = this.handlers[cmd];
    if (handler) {
      const response = handler(params);
      this.log.push({ seq: packet.seq, cmd, response, direction: "RX" });
      return { ok: true, seq: packet.seq, data: response };
    }
    return { ok: false, error: "Unknown: " + cmd };
  }
}

const proto = new CommandProtocol();

proto.register("MOVE_JOINT", (p) => {
  robot.setJoint(p.joint, p.angle);
  return { joint: p.joint, angle: p.angle, status: "OK" };
});
proto.register("SET_GRIPPER", (p) => {
  robot.setGripper(p.width / 80);
  return { width: p.width, force: 0, status: "OK" };
});
proto.register("GET_STATUS", () => {
  return { battery: 87, temp: 34.2, uptime: 3600, mode: "OPERATIONAL" };
});
proto.register("HOME", () => {
  robot.setJoint(0, 0); robot.setJoint(1, 0); robot.setJoint(2, 0);
  robot.setGripper(1);
  return { status: "HOMED" };
});

console.log("─── Command Protocol Demo ───");
let r;
r = proto.send("GET_STATUS", {});
console.log("STATUS: battery=" + r.data.battery + "% temp=" + r.data.temp + "°C");
r = proto.send("MOVE_JOINT", { joint: 1, angle: Math.PI/4 });
console.log("MOVE: joint=" + r.data.joint + " angle=" + (r.data.angle*180/Math.PI).toFixed(1) + "°");
r = proto.send("SET_GRIPPER", { width: 30 });
console.log("GRIPPER: width=" + r.data.width + "mm");
r = proto.send("HOME", {});
console.log("HOME: " + r.data.status);
console.log("Packets: " + proto.log.length + " (" + proto.log.filter(l=>l.direction==="TX").length + " TX, " + proto.log.filter(l=>l.direction==="RX").length + " RX)");
`,
    },
    {
      id: 'telemetry-stream', name: 'Telemetry Stream',
      description: 'High-frequency sensor data streaming with buffering',
      category: 'Communication', categoryId: 'communication', language: 'typescript',
      tags: ['telemetry', 'streaming', 'buffer'],
      code: `// Telemetry Stream — high-frequency sensor data with ring buffer

class RingBuffer {
  constructor(size) { this.buf = new Array(size); this.size = size; this.head = 0; this.count = 0; }
  push(item) { this.buf[this.head] = item; this.head = (this.head + 1) % this.size; this.count = Math.min(this.count + 1, this.size); }
  toArray() {
    const result = [];
    const start = this.count === this.size ? this.head : 0;
    for (let i = 0; i < this.count; i++) result.push(this.buf[(start + i) % this.size]);
    return result;
  }
  get length() { return this.count; }
}

class TelemetryStream {
  constructor(bufferSize) {
    this.channels = {};
    this.bufferSize = bufferSize;
    this.sampleCount = 0;
  }
  addChannel(name) { this.channels[name] = new RingBuffer(this.bufferSize); }
  record(name, value) {
    if (this.channels[name]) { this.channels[name].push({ t: this.sampleCount, v: value }); }
    this.sampleCount++;
  }
  getLatest(name, n) {
    const arr = this.channels[name]?.toArray() || [];
    return arr.slice(-n);
  }
  stats(name) {
    const arr = this.channels[name]?.toArray().map(s => s.v) || [];
    if (arr.length === 0) return null;
    const mean = arr.reduce((a,b) => a+b, 0) / arr.length;
    const variance = arr.reduce((a,b) => a + (b-mean)**2, 0) / arr.length;
    return { mean, std: Math.sqrt(variance), min: Math.min(...arr), max: Math.max(...arr), count: arr.length };
  }
}

const tel = new TelemetryStream(100);
tel.addChannel("joint_1_pos");
tel.addChannel("joint_1_vel");
tel.addChannel("motor_temp");
tel.addChannel("current_draw");

// Simulate 200 samples at 100Hz
for (let i = 0; i < 200; i++) {
  const t = i * 0.01;
  const pos = Math.sin(t * 2) * 0.5;
  const vel = Math.cos(t * 2) * 1.0;
  const temp = 25 + Math.abs(vel) * 5 + (Math.random()-0.5)*0.5;
  const current = 0.5 + Math.abs(vel) * 0.3 + (Math.random()-0.5)*0.05;
  
  tel.record("joint_1_pos", pos);
  tel.record("joint_1_vel", vel);
  tel.record("motor_temp", temp);
  tel.record("current_draw", current);
  
  robot.setJoint(1, pos);
}

console.log("─── Telemetry Statistics ───");
for (const ch of ["joint_1_pos", "joint_1_vel", "motor_temp", "current_draw"]) {
  const s = tel.stats(ch);
  console.log(ch + ": mean=" + s.mean.toFixed(3) + " std=" + s.std.toFixed(3) + " [" + s.min.toFixed(2) + ", " + s.max.toFixed(2) + "]");
}
console.log("Buffer: " + tel.getLatest("joint_1_pos", 1).length + " samples retained");
`,
    },
  ],
  safety: [
    {
      id: 'joint-limits', name: 'Joint Limits Monitor',
      description: 'Real-time limit monitoring with soft/hard stops',
      category: 'Safety', categoryId: 'safety', language: 'typescript',
      tags: ['limits', 'safety', 'monitor'],
      code: `// Joint Limits Monitor — soft/hard stops with emergency brake

class JointMonitor {
  constructor(limits) { this.limits = limits; this.violations = []; }
  check(name, angle) {
    const lim = this.limits[name];
    if (!lim) return { safe: true };
    const softMin = lim.min + lim.softMargin;
    const softMax = lim.max - lim.softMargin;
    
    if (angle < lim.min || angle > lim.max) {
      this.violations.push({ name, type: "HARD_LIMIT", angle });
      return { safe: false, type: "HARD_LIMIT", clampedAngle: Math.max(lim.min, Math.min(lim.max, angle)) };
    }
    if (angle < softMin || angle > softMax) {
      const scale = angle < softMin ? (angle - lim.min) / lim.softMargin : (lim.max - angle) / lim.softMargin;
      return { safe: true, type: "SOFT_LIMIT", velocityScale: scale };
    }
    return { safe: true, type: "NORMAL" };
  }
}

const monitor = new JointMonitor({
  base: { min: -Math.PI, max: Math.PI, softMargin: 0.2 },
  shoulder: { min: -Math.PI/2, max: Math.PI, softMargin: 0.15 },
  elbow: { min: -Math.PI, max: 0, softMargin: 0.15 },
});

const tests = [
  { name: "shoulder", angle: 0.5 },
  { name: "shoulder", angle: 2.8 },
  { name: "shoulder", angle: 3.5 },
  { name: "elbow", angle: -0.5 },
  { name: "elbow", angle: -2.9 },
  { name: "elbow", angle: -3.5 },
];

for (const test of tests) {
  const result = monitor.check(test.name, test.angle);
  const deg = (test.angle * 180 / Math.PI).toFixed(1);
  
  if (result.type === "HARD_LIMIT") {
    console.log("STOP " + test.name + " @ " + deg + "° — HARD LIMIT");
    robot.setJoint(1, result.clampedAngle);
  } else if (result.type === "SOFT_LIMIT") {
    console.log("WARN " + test.name + " @ " + deg + "° — vel scaled " + (result.velocityScale*100).toFixed(0) + "%");
    robot.setJoint(1, test.angle);
  } else {
    console.log("OK   " + test.name + " @ " + deg + "°");
    robot.setJoint(1, test.angle);
  }
}

console.log("Violations: " + monitor.violations.length);
`,
    },
    {
      id: 'watchdog-timer', name: 'Watchdog Timer',
      description: 'Communication timeout detection and safe shutdown',
      category: 'Safety', categoryId: 'safety', language: 'typescript',
      tags: ['watchdog', 'timeout', 'safety'],
      code: `// Watchdog Timer — communication timeout detection

class Watchdog {
  constructor(timeoutMs) {
    this.timeout = timeoutMs; this.lastFeed = Date.now();
    this.triggered = false; this.triggerCount = 0; this.callbacks = [];
  }
  feed() { this.lastFeed = Date.now(); this.triggered = false; }
  onTimeout(cb) { this.callbacks.push(cb); }
  check() {
    const elapsed = Date.now() - this.lastFeed;
    if (elapsed > this.timeout && !this.triggered) {
      this.triggered = true; this.triggerCount++;
      this.callbacks.forEach(cb => cb(elapsed));
      return { expired: true, elapsed };
    }
    return { expired: false, remaining: this.timeout - elapsed };
  }
}

const wd = new Watchdog(100);
let systemState = "RUNNING";

wd.onTimeout((elapsed) => {
  systemState = "SAFE_STOP";
  console.log("WATCHDOG TRIGGERED after " + elapsed + "ms");
  console.log("Executing safe shutdown...");
  robot.setJoint(0, 0); robot.setJoint(1, 0); robot.setJoint(2, 0);
  robot.setGripper(1);
});

const timeline = [
  { t: 0, action: "feed" },
  { t: 50, action: "feed" },
  { t: 90, action: "feed" },
  { t: 130, action: "feed" },
  { t: 160, action: "missed" },
  { t: 200, action: "missed" },
  { t: 300, action: "feed" },
];

for (const event of timeline) {
  if (event.action === "feed") {
    wd.feed();
    console.log("t=" + event.t + "ms: Feed OK — state=" + systemState);
    robot.setJoint(1, Math.PI/4);
  } else {
    const status = wd.check();
    console.log("t=" + event.t + "ms: " + (status.expired ? "TIMEOUT" : "OK " + status.remaining + "ms left") + " — state=" + systemState);
  }
}

console.log("Watchdog triggers: " + wd.triggerCount);
console.log("Final state: " + systemState);
`,
    },
    {
      id: 'collision-avoidance', name: 'Collision Avoidance',
      description: 'Potential field method for real-time obstacle avoidance',
      category: 'Safety', categoryId: 'safety', language: 'typescript',
      tags: ['collision', 'avoidance', 'potential-field'],
      code: `// Potential Field — real-time collision avoidance
// Attractive force toward goal + repulsive force from obstacles

function potentialField(pos, goal, obstacles) {
  // Attractive force
  const kAtt = 1.0;
  const dx = goal.x - pos.x, dy = goal.y - pos.y;
  const distGoal = Math.sqrt(dx*dx + dy*dy);
  const attX = kAtt * dx / (distGoal + 0.01);
  const attY = kAtt * dy / (distGoal + 0.01);
  
  // Repulsive forces
  const kRep = 0.5;
  const dInfluence = 0.8;
  let repX = 0, repY = 0;
  
  for (const obs of obstacles) {
    const ox = pos.x - obs.x, oy = pos.y - obs.y;
    const distObs = Math.sqrt(ox*ox + oy*oy) - obs.r;
    
    if (distObs < dInfluence && distObs > 0.01) {
      const mag = kRep * (1/distObs - 1/dInfluence) / (distObs * distObs);
      repX += mag * ox / Math.sqrt(ox*ox + oy*oy);
      repY += mag * oy / Math.sqrt(ox*ox + oy*oy);
    }
  }
  
  return { fx: attX + repX, fy: attY + repY };
}

const obstacles = [
  { x: 0.5, y: 0.5, r: 0.2 },
  { x: -0.3, y: 0.8, r: 0.15 },
  { x: 0.8, y: -0.2, r: 0.25 },
];

const goal = { x: 1.2, y: 0.8 };
let pos = { x: 0, y: 0 };
const stepSize = 0.05;
const path = [{ ...pos }];

for (let i = 0; i < 60; i++) {
  const { fx, fy } = potentialField(pos, goal, obstacles);
  const mag = Math.sqrt(fx*fx + fy*fy);
  
  if (mag > 0.01) {
    pos.x += (fx / mag) * stepSize;
    pos.y += (fy / mag) * stepSize;
    path.push({ ...pos });
  }
  
  const distToGoal = Math.sqrt((pos.x-goal.x)**2 + (pos.y-goal.y)**2);
  if (distToGoal < 0.1) {
    console.log("Goal reached in " + (i+1) + " steps!");
    break;
  }
}

robot.setJoint(1, Math.atan2(pos.y, pos.x));
robot.setJoint(2, -Math.PI/6);

console.log("Path length: " + path.length + " waypoints");
console.log("Start: (0, 0) → Goal: (" + goal.x + ", " + goal.y + ")");
console.log("Final: (" + pos.x.toFixed(2) + ", " + pos.y.toFixed(2) + ")");
console.log("Obstacles avoided: " + obstacles.length);
`,
    },
  ],
};

const CATEGORIES = [
  { id: 'control', label: 'Control', icon: Cpu, color: 'text-blue-400' },
  { id: 'perception', label: 'Perception', icon: Eye, color: 'text-red-400' },
  { id: 'hardware', label: 'Hardware', icon: Wrench, color: 'text-orange-400' },
  { id: 'simulation', label: 'Simulation', icon: Box, color: 'text-emerald-400' },
  { id: 'ai', label: 'AI / Agents', icon: Brain, color: 'text-purple-400' },
  { id: 'communication', label: 'Communication', icon: Radio, color: 'text-teal-400' },
  { id: 'safety', label: 'Safety', icon: ShieldCheck, color: 'text-yellow-400' },
];

interface SDKExplorerProps {
  onSelectModule: (module: SDKModule) => void;
  selectedModuleId?: string;
}

export const SDKExplorer = ({ onSelectModule, selectedModuleId }: SDKExplorerProps) => {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['control']));
  const [search, setSearch] = useState('');

  const filteredModules = useMemo(() => {
    if (!search.trim()) return SDK_MODULES;
    const q = search.toLowerCase();
    const result: Record<string, SDKModule[]> = {};
    for (const [catId, modules] of Object.entries(SDK_MODULES)) {
      const filtered = modules.filter(m =>
        m.name.toLowerCase().includes(q) ||
        m.description.toLowerCase().includes(q) ||
        m.tags.some(t => t.includes(q))
      );
      if (filtered.length > 0) result[catId] = filtered;
    }
    return result;
  }, [search]);

  const totalModules = Object.values(SDK_MODULES).reduce((s, m) => s + m.length, 0);

  const toggleCategory = (id: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="h-full flex flex-col bg-card/60 border-r border-border">
      {/* Header */}
      <div className="px-3 py-2.5 border-b border-border">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] font-semibold text-foreground uppercase tracking-wider">Explorer</span>
          <Badge variant="secondary" className="text-[10px] h-4 px-1.5 font-mono">{totalModules}</Badge>
        </div>
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
          <Input
            placeholder="Search modules..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-7 pl-7 text-[11px] bg-muted/30 border-border/50 focus:border-primary/50"
          />
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="py-1">
          {CATEGORIES.map(cat => {
            const Icon = cat.icon;
            const modules = filteredModules[cat.id] || [];
            if (search && modules.length === 0) return null;
            const isExpanded = expandedCategories.has(cat.id) || !!search;

            return (
              <div key={cat.id} className="mb-0.5">
                <button
                  onClick={() => toggleCategory(cat.id)}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-[11px] hover:bg-muted/30 transition-colors"
                >
                  {isExpanded ? (
                    <ChevronDown className="h-3 w-3 text-muted-foreground shrink-0" />
                  ) : (
                    <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />
                  )}
                  <Icon className={`h-3.5 w-3.5 shrink-0 ${cat.color}`} />
                  <span className="font-medium text-foreground/90">{cat.label}</span>
                  <span className="ml-auto text-[10px] text-muted-foreground">{modules.length}</span>
                </button>

                {isExpanded && (
                  <div className="ml-3 border-l border-border/50">
                    {modules.map(mod => (
                      <button
                        key={mod.id}
                        onClick={() => onSelectModule(mod)}
                        className={`w-full flex items-start gap-2 pl-4 pr-3 py-1.5 text-left transition-colors ${
                          selectedModuleId === mod.id
                            ? 'bg-primary/10 border-l-2 border-primary -ml-px'
                            : 'hover:bg-muted/20 text-muted-foreground hover:text-foreground'
                        }`}
                      >
                        <FileCode className="h-3 w-3 mt-0.5 shrink-0 opacity-50" />
                        <div className="min-w-0">
                          <div className={`text-[11px] font-medium truncate ${selectedModuleId === mod.id ? 'text-primary' : ''}`}>
                            {mod.name}
                          </div>
                          <div className="text-[10px] text-muted-foreground/60 leading-tight mt-0.5 line-clamp-1">
                            {mod.description}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>

      {/* Footer */}
      <div className="px-3 py-2 border-t border-border">
        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground/50">
          <Sparkles className="h-3 w-3" />
          <span>{CATEGORIES.length} categories · {totalModules} modules</span>
        </div>
      </div>
    </div>
  );
};

export { SDK_MODULES };
