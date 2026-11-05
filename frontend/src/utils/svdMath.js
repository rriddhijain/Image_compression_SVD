/**
 * Client-side SVD math utilities for 2D matrix reconstruction.
 * Uses power iteration with deflation for fast, zero-dependency client-side SVD.
 */

// Compute top-k singular value decomposition using power iteration with Gram-Schmidt deflation
export function computeTopKSVD(matrix, k, maxIter = 25) {
  const m = matrix.length;
  const n = matrix[0].length;
  const effectiveK = Math.min(k, Math.min(m, n));

  const U = [];
  const S = [];
  const V = [];

  // Work on a copy of the matrix
  let A = matrix.map(row => [...row]);

  for (let comp = 0; comp < effectiveK; comp++) {
    // Random initial vector
    let v = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      v[j] = Math.sin((comp + 1) * 997 + j * 13);
    }
    
    // Normalize v
    let normV = 0;
    for (let j = 0; j < n; j++) normV += v[j] * v[j];
    normV = Math.sqrt(normV) || 1e-12;
    for (let j = 0; j < n; j++) v[j] /= normV;

    let u = new Float64Array(m);
    let sigma = 0;

    // Power iterations
    for (let iter = 0; iter < maxIter; iter++) {
      // u = A * v
      for (let i = 0; i < m; i++) {
        let sum = 0;
        const row = A[i];
        for (let j = 0; j < n; j++) {
          sum += row[j] * v[j];
        }
        u[i] = sum;
      }

      // sigma = ||u||
      let normU = 0;
      for (let i = 0; i < m; i++) normU += u[i] * u[i];
      sigma = Math.sqrt(normU);
      if (sigma < 1e-12) break;

      // normalize u
      for (let i = 0; i < m; i++) u[i] /= sigma;

      // v = A^T * u
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let i = 0; i < m; i++) {
          sum += A[i][j] * u[i];
        }
        v[j] = sum;
      }

      // normalize v
      normV = 0;
      for (let j = 0; j < n; j++) normV += v[j] * v[j];
      normV = Math.sqrt(normV) || 1e-12;
      for (let j = 0; j < n; j++) v[j] /= normV;
    }

    // Save singular triplet
    U.push(Array.from(u));
    S.push(sigma);
    V.push(Array.from(v));

    // Deflate: A = A - sigma * u * v^T
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        A[i][j] -= sigma * u[i] * v[j];
      }
    }
  }

  return { U, S, V };
}

// Reconstruct matrix from top-k triplets
export function reconstructFromSVD(U, S, V, k) {
  const m = U[0].length;
  const n = V[0].length;
  const effectiveK = Math.min(k, S.length);

  const result = Array.from({ length: m }, () => new Float64Array(n));

  for (let comp = 0; comp < effectiveK; comp++) {
    const sigma = S[comp];
    const u = U[comp];
    const v = V[comp];

    for (let i = 0; i < m; i++) {
      const uVal = u[i] * sigma;
      for (let j = 0; j < n; j++) {
        result[i][j] += uVal * v[j];
      }
    }
  }

  return result;
}

// Compute Frobenius norm error and cumulative energy
export function calculateSVDMetrics(origMatrix, reconMatrix, S, k) {
  const m = origMatrix.length;
  const n = origMatrix[0].length;

  let origFrobSq = 0;
  let diffFrobSq = 0;

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const orig = origMatrix[i][j];
      const recon = reconMatrix[i][j];
      origFrobSq += orig * orig;
      const diff = orig - recon;
      diffFrobSq += diff * diff;
    }
  }

  const origFrob = Math.sqrt(origFrobSq) || 1e-12;
  const diffFrob = Math.sqrt(diffFrobSq);
  const relError = (diffFrob / origFrob) * 100;

  // Energy
  const totalEnergy = S.reduce((acc, s) => acc + s, 0) || 1;
  const retainedEnergy = S.slice(0, k).reduce((acc, s) => acc + s, 0);
  const energyPercent = Math.min(100, (retainedEnergy / totalEnergy) * 100);

  // Compression ratio
  const origParams = m * n;
  const decompParams = k * (m + n + 1);
  const compressionRatio = (origParams / decompParams).toFixed(2);

  return {
    relError: relError.toFixed(2),
    energyPercent: energyPercent.toFixed(2),
    compressionRatio,
    origParams,
    decompParams
  };
}
