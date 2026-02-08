// main.js - Gravitational Light Simulator
// Three.js visualization of variable speed of light in Schwarzschild spacetime

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

import {
    C, G, M_SUN,
    schwarzschildRadius, photonSphereRadius,
    effectiveLightSpeed, refractiveIndex, speedRatio,
    traceRay, deflectionAngle, shapiroDelay,
    solarMassesToKg, formatSci, radiansToArcseconds, radiansToDegrees,
    getPresetProperties
} from './physics.js';

// ============================================================
// Global State
// ============================================================

const state = {
    mass: 10.0,                    // Solar masses
    impactParameter: 5.0,          // In units of r_s
    showHeatmap: true,
    showPhotonSphere: true,
    showEventHorizon: true,
    rays: [],                      // Active ray traces
    animatingPhotons: [],          // Photons currently animating
    visualScale: 1.0,              // pixels per r_s
    bodyColor: 0x111111,
    bodyEmissive: 0x220000,
    preset: 'black_hole'
};

// ============================================================
// Scene Setup
// ============================================================

let scene, camera, renderer, controls;
let centralBody, eventHorizonMesh, photonSphereMesh;
let heatmapMesh, gridHelper;
let raycaster, mouse;
let clock;

// DOM references
let viewport, infoPanel;

function init() {
    clock = new THREE.Clock();
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    viewport = document.getElementById('viewport');

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020208);
    scene.fog = new THREE.FogExp2(0x020208, 0.003);

    // Camera
    const aspect = viewport.clientWidth / viewport.clientHeight;
    camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 2000);
    camera.position.set(0, 60, 80);
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(viewport.clientWidth, viewport.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    viewport.appendChild(renderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 5;
    controls.maxDistance = 500;
    controls.maxPolarAngle = Math.PI;

    // Lighting
    const ambient = new THREE.AmbientLight(0x222244, 0.5);
    scene.add(ambient);

    const pointLight = new THREE.PointLight(0xffffff, 1.5, 300);
    pointLight.position.set(50, 50, 50);
    scene.add(pointLight);

    // Star field background
    createStarField();

    // Grid
    gridHelper = new THREE.GridHelper(200, 40, 0x1a1a3a, 0x0d0d1f);
    gridHelper.position.y = -0.1;
    scene.add(gridHelper);

    // Create scene objects
    createCentralBody();
    createEventHorizon();
    createPhotonSphere();
    createHeatmap();

    // Axes helper (small, subtle)
    const axesSize = 3;
    const axesHelper = new THREE.AxesHelper(axesSize);
    axesHelper.position.set(-95, 0.1, -95);
    scene.add(axesHelper);

    // Event listeners
    window.addEventListener('resize', onResize);
    viewport.addEventListener('mousemove', onMouseMove);
    viewport.addEventListener('click', onViewportClick);

    // Bind UI
    bindControls();

    // Update scene for initial state
    updateScene();

    // Start animation loop
    animate();
}

// ============================================================
// Star Field
// ============================================================

function createStarField() {
    const starsGeometry = new THREE.BufferGeometry();
    const starCount = 3000;
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);
    const sizes = new Float32Array(starCount);

    for (let i = 0; i < starCount; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = 400 + Math.random() * 200;

        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = r * Math.cos(phi);

        const brightness = 0.5 + Math.random() * 0.5;
        const tint = Math.random();
        colors[i * 3] = brightness * (0.8 + tint * 0.2);
        colors[i * 3 + 1] = brightness * (0.8 + tint * 0.1);
        colors[i * 3 + 2] = brightness;

        sizes[i] = 0.5 + Math.random() * 1.5;
    }

    starsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    starsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    starsGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const starsMaterial = new THREE.PointsMaterial({
        size: 1.0,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        sizeAttenuation: false
    });

    const stars = new THREE.Points(starsGeometry, starsMaterial);
    scene.add(stars);
}

// ============================================================
// Central Body
// ============================================================

function createCentralBody() {
    const geometry = new THREE.SphereGeometry(1, 64, 64);
    const material = new THREE.MeshStandardMaterial({
        color: state.bodyColor,
        emissive: state.bodyEmissive,
        emissiveIntensity: 1.5,
        roughness: 0.3,
        metalness: 0.1
    });
    centralBody = new THREE.Mesh(geometry, material);
    scene.add(centralBody);

    // Glow effect using a sprite
    const glowCanvas = createGlowTexture();
    const glowTexture = new THREE.CanvasTexture(glowCanvas);
    const glowMaterial = new THREE.SpriteMaterial({
        map: glowTexture,
        transparent: true,
        blending: THREE.AdditiveBlending,
        opacity: 0.6,
        depthWrite: false
    });
    const glow = new THREE.Sprite(glowMaterial);
    glow.scale.set(8, 8, 1);
    glow.name = 'bodyGlow';
    centralBody.add(glow);
}

function createGlowTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    const gradient = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
    gradient.addColorStop(0, 'rgba(255, 200, 100, 0.8)');
    gradient.addColorStop(0.3, 'rgba(255, 100, 50, 0.4)');
    gradient.addColorStop(0.6, 'rgba(100, 50, 200, 0.1)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 256, 256);
    return canvas;
}

// ============================================================
// Event Horizon
// ============================================================

function createEventHorizon() {
    const geometry = new THREE.SphereGeometry(1, 48, 48);
    const material = new THREE.MeshBasicMaterial({
        color: 0x000000,
        transparent: true,
        opacity: 0.95,
        side: THREE.DoubleSide
    });
    eventHorizonMesh = new THREE.Mesh(geometry, material);
    eventHorizonMesh.renderOrder = -1;
    scene.add(eventHorizonMesh);
}

// ============================================================
// Photon Sphere
// ============================================================

function createPhotonSphere() {
    const geometry = new THREE.SphereGeometry(1, 64, 32);
    const material = new THREE.MeshBasicMaterial({
        color: 0xffaa00,
        wireframe: true,
        transparent: true,
        opacity: 0.25,
        depthWrite: false
    });
    photonSphereMesh = new THREE.Mesh(geometry, material);
    scene.add(photonSphereMesh);
}

// ============================================================
// Speed Heatmap
// ============================================================

function createHeatmap() {
    const size = 200;
    const segments = 200;
    const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
    geometry.rotateX(-Math.PI / 2);

    // We will update vertex colors in updateHeatmap()
    const colors = new Float32Array(geometry.attributes.position.count * 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide,
        depthWrite: false
    });

    heatmapMesh = new THREE.Mesh(geometry, material);
    heatmapMesh.position.y = -0.05;
    heatmapMesh.name = 'heatmap';
    scene.add(heatmapMesh);

    updateHeatmap();
}

function updateHeatmap() {
    if (!heatmapMesh) return;

    const M = solarMassesToKg(state.mass);
    const rs = schwarzschildRadius(M);
    const vs = state.visualScale;

    const geometry = heatmapMesh.geometry;
    const positions = geometry.attributes.position.array;
    const colors = geometry.attributes.color.array;

    for (let i = 0; i < positions.length / 3; i++) {
        const x = positions[i * 3];
        const z = positions[i * 3 + 2];
        const dist = Math.sqrt(x * x + z * z);

        // Convert visual distance to physical distance
        const r = dist / vs;

        // Compute c_eff / c ratio
        let ratio;
        if (r <= rs) {
            ratio = 0;
        } else {
            ratio = 1 - rs / r;
        }

        // Map ratio to color: blue (1.0 = speed of c) -> red (0 = stopped)
        const color = speedToColor(ratio);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
    }

    geometry.attributes.color.needsUpdate = true;
}

/**
 * Maps a speed ratio (0 to 1) to a color.
 * 0 (stopped) = deep red/black
 * 0.5 = orange/yellow
 * 1.0 (full c) = blue/cyan
 */
function speedToColor(ratio) {
    ratio = Math.max(0, Math.min(1, ratio));

    let r, g, b;

    if (ratio < 0.25) {
        // Black to red
        const t = ratio / 0.25;
        r = t * 0.8;
        g = 0;
        b = 0;
    } else if (ratio < 0.5) {
        // Red to orange/yellow
        const t = (ratio - 0.25) / 0.25;
        r = 0.8 + t * 0.2;
        g = t * 0.6;
        b = 0;
    } else if (ratio < 0.75) {
        // Yellow to cyan
        const t = (ratio - 0.5) / 0.25;
        r = 1.0 - t * 0.8;
        g = 0.6 + t * 0.4;
        b = t * 0.8;
    } else {
        // Cyan to blue
        const t = (ratio - 0.75) / 0.25;
        r = 0.2 - t * 0.15;
        g = 1.0 - t * 0.3;
        b = 0.8 + t * 0.2;
    }

    return { r, g, b };
}

// ============================================================
// Ray Tracing & Visualization
// ============================================================

function fireRay() {
    const M = solarMassesToKg(state.mass);
    const rs = schwarzschildRadius(M);
    const vs = state.visualScale;

    // Impact parameter in physical units
    const b = state.impactParameter * rs;

    // Start position: far away, offset by impact parameter
    const startDistance = 80 / vs; // physical distance corresponding to 80 visual units
    const startX = -startDistance;
    const startY = b;

    // Direction: along +x
    const dir = { x: 1, y: 0 };

    // Trace the ray
    const result = traceRay(
        { x: startX * vs, y: startY * vs },
        dir,
        M,
        20000,      // max steps
        0.0003,     // dt (smaller for accuracy)
        vs
    );

    // Create visual ray
    const rayGroup = new THREE.Group();
    rayGroup.name = 'lightRay';

    if (result.points.length > 1) {
        // Build line geometry with colors
        const linePoints = [];
        const lineColors = [];

        for (let i = 0; i < result.points.length; i++) {
            const p = result.points[i];
            linePoints.push(new THREE.Vector3(p.x, 0.2, p.y));

            // Color based on local speed
            const speedFrac = (result.speeds[i] || 0) / C;
            const col = speedToColor(speedFrac);
            lineColors.push(col.r, col.g, col.b);
        }

        // Line
        const lineGeom = new THREE.BufferGeometry().setFromPoints(linePoints);
        lineGeom.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

        const lineMat = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            linewidth: 2
        });

        const line = new THREE.Line(lineGeom, lineMat);
        rayGroup.add(line);

        // Also create a "tube" version for visual thickness using a series of tiny spheres at key points
        // (Three.js line width is limited to 1px in WebGL, so we add glow dots)
        const dotSpacing = Math.max(1, Math.floor(result.points.length / 150));
        for (let i = 0; i < result.points.length; i += dotSpacing) {
            const p = result.points[i];
            const speedFrac = (result.speeds[i] || 0) / C;
            const col = speedToColor(speedFrac);

            const dotGeom = new THREE.SphereGeometry(0.15, 6, 6);
            const dotMat = new THREE.MeshBasicMaterial({
                color: new THREE.Color(col.r, col.g, col.b),
                transparent: true,
                opacity: 0.4
            });
            const dot = new THREE.Mesh(dotGeom, dotMat);
            dot.position.set(p.x, 0.2, p.y);
            rayGroup.add(dot);
        }
    }

    scene.add(rayGroup);
    state.rays.push(rayGroup);

    // Create animated photon
    createPhoton(result);

    // Update info display
    displayRayInfo(result);
}

function createPhoton(rayResult) {
    // Photon light
    const photonGroup = new THREE.Group();

    // Glowing sphere
    const photonGeom = new THREE.SphereGeometry(0.5, 16, 16);
    const photonMat = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.95
    });
    const photonMesh = new THREE.Mesh(photonGeom, photonMat);
    photonGroup.add(photonMesh);

    // Point light attached to photon
    const photonLight = new THREE.PointLight(0xaaccff, 2, 15);
    photonGroup.add(photonLight);

    // Glow sprite
    const glowCanvas = document.createElement('canvas');
    glowCanvas.width = 64;
    glowCanvas.height = 64;
    const ctx = glowCanvas.getContext('2d');
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(180, 220, 255, 1.0)');
    gradient.addColorStop(0.3, 'rgba(100, 150, 255, 0.5)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);

    const glowTexture = new THREE.CanvasTexture(glowCanvas);
    const glowSprite = new THREE.Sprite(
        new THREE.SpriteMaterial({
            map: glowTexture,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        })
    );
    glowSprite.scale.set(3, 3, 1);
    photonGroup.add(glowSprite);

    if (rayResult.points.length > 0) {
        const p = rayResult.points[0];
        photonGroup.position.set(p.x, 0.2, p.y);
    }

    scene.add(photonGroup);

    state.animatingPhotons.push({
        group: photonGroup,
        result: rayResult,
        currentIndex: 0,
        time: 0,
        speed: 1.0
    });
}

function animatePhotons(delta) {
    const toRemove = [];

    for (let i = 0; i < state.animatingPhotons.length; i++) {
        const photon = state.animatingPhotons[i];
        const result = photon.result;

        // Move along the path - speed proportional to local c_eff
        const speedFrac = (result.speeds[photon.currentIndex] || 0.5) / C;
        const moveRate = 2 + speedFrac * 6; // base rate + speed-dependent rate

        photon.time += delta * moveRate * 60; // 60 fps normalization

        const idx = Math.floor(photon.time);

        if (idx >= result.points.length - 1) {
            // Photon has finished its path
            toRemove.push(i);
            continue;
        }

        photon.currentIndex = idx;

        // Interpolate position
        const frac = photon.time - idx;
        const p1 = result.points[idx];
        const p2 = result.points[Math.min(idx + 1, result.points.length - 1)];

        const x = p1.x + (p2.x - p1.x) * frac;
        const y = p1.y + (p2.y - p1.y) * frac;

        photon.group.position.set(x, 0.2, y);

        // Update photon color based on local speed
        const spdFrac = (result.speeds[idx] || 0) / C;
        const col = speedToColor(spdFrac);
        const mesh = photon.group.children[0];
        if (mesh && mesh.material) {
            mesh.material.color.setRGB(
                0.5 + col.r * 0.5,
                0.5 + col.g * 0.5,
                0.7 + col.b * 0.3
            );
        }

        // Pulse the glow
        const pulse = 1.0 + 0.3 * Math.sin(photon.time * 0.5);
        const sprite = photon.group.children[2];
        if (sprite) {
            sprite.scale.set(3 * pulse, 3 * pulse, 1);
        }
    }

    // Remove finished photons (iterate in reverse)
    for (let i = toRemove.length - 1; i >= 0; i--) {
        const idx = toRemove[i];
        const photon = state.animatingPhotons[idx];
        scene.remove(photon.group);
        photon.group.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        });
        state.animatingPhotons.splice(idx, 1);
    }
}

function clearRays() {
    // Remove ray lines
    for (const ray of state.rays) {
        scene.remove(ray);
        ray.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
    }
    state.rays = [];

    // Remove photons
    for (const photon of state.animatingPhotons) {
        scene.remove(photon.group);
        photon.group.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        });
    }
    state.animatingPhotons = [];

    // Clear info
    document.getElementById('ray-info').innerHTML = '<em>Fire a ray to see results</em>';
}

// ============================================================
// Info Display
// ============================================================

function displayRayInfo(result) {
    const M = solarMassesToKg(state.mass);
    const rs = schwarzschildRadius(M);

    let html = '';

    if (result.captured) {
        html += '<div class="info-captured">RAY CAPTURED BY BLACK HOLE</div>';
    }

    html += `<div class="info-row">
        <span class="info-label">Deflection:</span>
        <span class="info-value">${radiansToDegrees(result.deflection).toFixed(4)}&deg; (${radiansToArcseconds(result.deflection).toFixed(2)}")</span>
    </div>`;

    html += `<div class="info-row">
        <span class="info-label">Impact param:</span>
        <span class="info-value">${(result.impactParameter / rs).toFixed(2)} r<sub>s</sub> (${formatSci(result.impactParameter)} m)</span>
    </div>`;

    html += `<div class="info-row">
        <span class="info-label">Min approach:</span>
        <span class="info-value">${(result.minDistance / rs).toFixed(2)} r<sub>s</sub> (${formatSci(result.minDistance)} m)</span>
    </div>`;

    // Analytical weak-field deflection for comparison
    if (result.impactParameter > 0) {
        const analytical = deflectionAngle(result.impactParameter, M);
        html += `<div class="info-row">
            <span class="info-label">Analytical:</span>
            <span class="info-value">${radiansToDegrees(analytical).toFixed(4)}&deg;</span>
        </div>`;
    }

    html += `<div class="info-row">
        <span class="info-label">Path points:</span>
        <span class="info-value">${result.points.length}</span>
    </div>`;

    html += `<div class="info-row">
        <span class="info-label">Schwarzschild r:</span>
        <span class="info-value">${formatSci(rs)} m</span>
    </div>`;

    document.getElementById('ray-info').innerHTML = html;
}

function updateCursorInfo(r, M) {
    const rs = schwarzschildRadius(M);
    const el = document.getElementById('cursor-info');

    if (r <= rs) {
        el.innerHTML = `r = ${(r / rs).toFixed(2)} r<sub>s</sub> | <strong>Inside event horizon</strong>`;
        return;
    }

    const cRatio = speedRatio(r, M);
    const n = refractiveIndex(r, M);
    const cEff = effectiveLightSpeed(r, M, 'radial');

    el.innerHTML = `r = ${(r / rs).toFixed(2)} r<sub>s</sub> | c<sub>eff</sub>/c = ${cRatio.toFixed(4)} | n = ${n.toFixed(4)} | c<sub>eff</sub> = ${formatSci(cEff)} m/s`;
}

// ============================================================
// Scene Updates
// ============================================================

function updateScene() {
    const M = solarMassesToKg(state.mass);
    const rs = schwarzschildRadius(M);
    const rps = photonSphereRadius(M);

    // Compute visual scale so that the Schwarzschild radius maps to a
    // reasonable visual size. For a 10 M_sun BH, rs ~ 29.5 km.
    // We want rs to appear as ~2 visual units at 10 M_sun.
    const targetVisualRs = 2 + Math.log10(state.mass + 1) * 1.5;
    state.visualScale = targetVisualRs / rs;
    const vs = state.visualScale;

    // Update central body
    const bodyVisualRadius = Math.max(0.5, targetVisualRs * 0.8);
    centralBody.scale.setScalar(bodyVisualRadius);

    // Update body appearance based on preset
    const preset = getPresetProperties(state.preset);
    centralBody.material.color.setHex(preset.color);
    centralBody.material.emissive.setHex(preset.emissiveColor);

    // Update glow
    const glow = centralBody.getObjectByName('bodyGlow');
    if (glow) {
        const glowScale = bodyVisualRadius * 4;
        glow.scale.set(glowScale / bodyVisualRadius, glowScale / bodyVisualRadius, 1);
        if (state.preset === 'black_hole') {
            glow.material.opacity = 0.15;
        } else if (state.preset === 'neutron_star') {
            glow.material.opacity = 0.7;
        } else {
            glow.material.opacity = 0.5;
        }
    }

    // Update event horizon
    const ehVisualRadius = targetVisualRs;
    eventHorizonMesh.scale.setScalar(ehVisualRadius);
    eventHorizonMesh.visible = state.showEventHorizon;

    // Update photon sphere
    const psVisualRadius = 1.5 * targetVisualRs;
    photonSphereMesh.scale.setScalar(psVisualRadius);
    photonSphereMesh.visible = state.showPhotonSphere;

    // Update heatmap
    heatmapMesh.visible = state.showHeatmap;
    if (state.showHeatmap) {
        updateHeatmap();
    }

    // Update display values
    updateDisplayValues();
}

function updateDisplayValues() {
    const M = solarMassesToKg(state.mass);
    const rs = schwarzschildRadius(M);
    const rps = photonSphereRadius(M);

    document.getElementById('mass-value').textContent = state.mass.toFixed(2);
    document.getElementById('impact-value').textContent = state.impactParameter.toFixed(1);

    document.getElementById('rs-display').textContent = formatSci(rs) + ' m';
    document.getElementById('rps-display').textContent = formatSci(rps) + ' m';

    // Weak-field deflection for current impact parameter
    const b = state.impactParameter * rs;
    if (b > 0) {
        const defl = deflectionAngle(b, M);
        document.getElementById('defl-display').textContent =
            radiansToDegrees(defl).toFixed(4) + '\u00B0';
    }
}

// ============================================================
// Mouse Interaction
// ============================================================

function onMouseMove(event) {
    const rect = viewport.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Raycast to the heatmap plane
    raycaster.setFromCamera(mouse, camera);

    // Intersect with a horizontal plane at y=0
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const intersection = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, intersection);

    if (intersection) {
        const dist = Math.sqrt(intersection.x * intersection.x + intersection.z * intersection.z);
        const M = solarMassesToKg(state.mass);
        const physicalDist = dist / state.visualScale;
        updateCursorInfo(physicalDist, M);
    }
}

function onViewportClick(event) {
    // Same as mousemove but also display info prominently
    onMouseMove(event);
}

// ============================================================
// UI Bindings
// ============================================================

function bindControls() {
    // Mass slider (logarithmic)
    const massSlider = document.getElementById('mass-slider');
    massSlider.addEventListener('input', (e) => {
        // Logarithmic mapping: 0-100 -> 0.1 to 100 solar masses
        const t = parseFloat(e.target.value) / 100;
        state.mass = 0.1 * Math.pow(1000, t);
        updateScene();
    });

    // Impact parameter slider
    const impactSlider = document.getElementById('impact-slider');
    impactSlider.addEventListener('input', (e) => {
        state.impactParameter = parseFloat(e.target.value);
        updateDisplayValues();
    });

    // Fire ray button
    document.getElementById('btn-fire').addEventListener('click', fireRay);

    // Clear rays button
    document.getElementById('btn-clear').addEventListener('click', clearRays);

    // Toggle heatmap
    document.getElementById('chk-heatmap').addEventListener('change', (e) => {
        state.showHeatmap = e.target.checked;
        updateScene();
    });

    // Toggle photon sphere
    document.getElementById('chk-photon-sphere').addEventListener('change', (e) => {
        state.showPhotonSphere = e.target.checked;
        updateScene();
    });

    // Toggle event horizon
    document.getElementById('chk-event-horizon').addEventListener('change', (e) => {
        state.showEventHorizon = e.target.checked;
        updateScene();
    });

    // Presets
    document.getElementById('preset-sun').addEventListener('click', () => {
        state.mass = 1.0;
        state.preset = 'sun';
        document.getElementById('mass-slider').value = 33.3; // log scale position for 1.0
        state.impactParameter = 50;
        document.getElementById('impact-slider').value = 50;
        updateScene();
    });

    document.getElementById('preset-neutron').addEventListener('click', () => {
        state.mass = 2.0;
        state.preset = 'neutron_star';
        document.getElementById('mass-slider').value = 43.7;
        state.impactParameter = 15;
        document.getElementById('impact-slider').value = 15;
        updateScene();
    });

    document.getElementById('preset-bh').addEventListener('click', () => {
        state.mass = 10.0;
        state.preset = 'black_hole';
        document.getElementById('mass-slider').value = 66.7;
        state.impactParameter = 5;
        document.getElementById('impact-slider').value = 5;
        updateScene();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'f' || e.key === 'F') fireRay();
        if (e.key === 'c' || e.key === 'C') clearRays();
    });
}

// ============================================================
// Resize Handler
// ============================================================

function onResize() {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// ============================================================
// Animation Loop
// ============================================================

function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    // Update controls
    controls.update();

    // Animate photons
    animatePhotons(delta);

    // Subtle rotation of photon sphere wireframe
    if (photonSphereMesh.visible) {
        photonSphereMesh.rotation.y += delta * 0.1;
        photonSphereMesh.rotation.x = Math.sin(Date.now() * 0.0003) * 0.05;
    }

    // Render
    renderer.render(scene, camera);
}

// ============================================================
// Initialize
// ============================================================

init();
