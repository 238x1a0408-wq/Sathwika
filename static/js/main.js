// ============================================================
//  Smart Lender - Main JavaScript
//  Handles charts, predictions, batch upload, and UI effects
// ============================================================

// ── Utility ──────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function showToast(message, type = 'info') {
  const container = $('#toast-container') || (() => {
    const el = document.createElement('div');
    el.id = 'toast-container';
    el.className = 'toast-container';
    document.body.appendChild(el);
    return el;
  })();

  const icons = { info: '💡', success: '✅', error: '❌', warning: '⚠️' };
  const colors = { info: 'var(--indigo)', success: 'var(--green)', error: 'var(--red)', warning: 'var(--amber)' };

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.style.borderLeftColor = colors[type];
  toast.style.borderLeftWidth = '3px';
  toast.style.borderLeftStyle = 'solid';
  toast.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function animateNumber(el, target, duration = 1200, suffix = '') {
  const start  = 0;
  const step   = (timestamp) => {
    if (!startTime) startTime = timestamp;
    const progress = Math.min((timestamp - startTime) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.floor(eased * target) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  };
  let startTime = null;
  requestAnimationFrame(step);
}

// ── Active Nav Link ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;
  $$('.nav-links a').forEach(a => {
    if (a.getAttribute('href') === path || (path === '/' && a.getAttribute('href') === '/')) {
      a.classList.add('active');
    }
  });

  // Animate stat numbers on page load
  $$('[data-count]').forEach(el => {
    const target  = parseFloat(el.dataset.count);
    const suffix  = el.dataset.suffix || '';
    const isFloat = el.dataset.float === 'true';
    if (!isFloat) {
      animateNumber(el, target, 1400, suffix);
    } else {
      let startTime = null;
      const step = (ts) => {
        if (!startTime) startTime = ts;
        const p = Math.min((ts - startTime) / 1400, 1);
        const e = 1 - Math.pow(1 - p, 3);
        el.textContent = (e * target).toFixed(1) + suffix;
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }
  });

  // Animate metric bars
  $$('.metric-bar[data-width]').forEach(bar => {
    setTimeout(() => {
      bar.style.width = bar.dataset.width + '%';
    }, 200);
  });

  initCharts();
});

// ── Model Comparison Charts ───────────────────────────────────
function initCharts() {
  const ctx = $('#modelCompareChart');
  if (!ctx || typeof Chart === 'undefined') return;

  const models  = window.SMART_LENDER?.models || {};
  const labels  = Object.keys(models);
  const trainAcc = labels.map(m => models[m].train_accuracy);
  const testAcc  = labels.map(m => models[m].test_accuracy);
  const cvAcc    = labels.map(m => models[m].cv_accuracy);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Training Accuracy %',
          data: trainAcc,
          backgroundColor: 'rgba(99,102,241,0.75)',
          borderRadius: 8,
          borderSkipped: false,
        },
        {
          label: 'Test Accuracy %',
          data: testAcc,
          backgroundColor: 'rgba(16,185,129,0.75)',
          borderRadius: 8,
          borderSkipped: false,
        },
        {
          label: 'CV Accuracy %',
          data: cvAcc,
          backgroundColor: 'rgba(139,92,246,0.60)',
          borderRadius: 8,
          borderSkipped: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#94A3B8',
            font: { family: 'Inter', size: 12 },
            boxWidth: 14,
            padding: 20,
          }
        },
        tooltip: {
          backgroundColor: 'rgba(14,20,32,0.95)',
          titleColor: '#F1F5F9',
          bodyColor: '#94A3B8',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y}%`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#94A3B8', font: { family: 'Inter', size: 12 } },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          min: 60,
          max: 100,
          ticks: {
            color: '#94A3B8',
            font: { family: 'Inter', size: 11 },
            callback: (v) => v + '%'
          },
          grid: { color: 'rgba(255,255,255,0.06)' },
        }
      }
    }
  });

  // Feature Importance doughnut chart
  const featureCtx = $('#featureImportanceChart');
  if (!featureCtx) return;

  const fi     = window.SMART_LENDER?.feature_importance || {};
  const fiKeys = Object.keys(fi).sort((a, b) => fi[b] - fi[a]).slice(0, 6);
  const fiVals = fiKeys.map(k => (fi[k] * 100).toFixed(1));
  const palette = ['#8B5CF6', '#6366F1', '#06B6D4', '#10B981', '#F59E0B', '#EF4444'];

  new Chart(featureCtx, {
    type: 'doughnut',
    data: {
      labels: fiKeys,
      datasets: [{
        data: fiVals,
        backgroundColor: palette,
        borderColor: 'transparent',
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: {
          position: 'right',
          labels: {
            color: '#94A3B8',
            font: { family: 'Inter', size: 12 },
            padding: 14,
            boxWidth: 12,
          }
        },
        tooltip: {
          backgroundColor: 'rgba(14,20,32,0.95)',
          titleColor: '#F1F5F9',
          bodyColor: '#94A3B8',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          callbacks: { label: (ctx) => ` ${ctx.label}: ${ctx.raw}%` }
        }
      }
    }
  });
}

// ── Prediction Form ───────────────────────────────────────────
const predictForm = $('#predictForm');
if (predictForm) {
  predictForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const submitBtn    = $('#submitPrediction');
    const resultPanel  = $('#resultPanel');
    const resultContent= $('#resultContent');

    // Show loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<div class="spinner" style="width:20px;height:20px;border-width:2px;"></div> Analyzing...';
    resultContent.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:16px;">
        <div class="spinner"></div>
        <p style="color:var(--text-secondary);font-size:14px;">Processing applicant data...</p>
      </div>`;

    const formData = new FormData(predictForm);
    const payload  = Object.fromEntries(formData.entries());

    try {
      const response = await fetch('/api/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || 'Prediction failed');
      }

      displayPredictionResult(data, resultContent);
      showToast(data.approved ? '✅ Loan Approved with high confidence!' : '⚠️ Loan Rejected — flagged for review.', data.approved ? 'success' : 'warning');

    } catch (err) {
      resultContent.innerHTML = `<div class="result-placeholder"><div class="icon">⚠️</div><p style="color:var(--red-light)">${err.message}</p></div>`;
      showToast('Prediction failed: ' + err.message, 'error');
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<span>🔍</span> Analyze Application';
    }
  });
}

function displayPredictionResult(data, container) {
  const icon       = data.approved ? '✅' : '❌';
  const verdictCls = data.approved ? 'verdict-approved' : 'verdict-rejected';
  const riskCls    = { green: 'risk-green', amber: 'risk-amber', red: 'risk-red' }[data.risk_color] || 'risk-amber';
  const riskIcon   = { green: '🟢', amber: '🟡', red: '🔴' }[data.risk_color] || '🟡';

  container.innerHTML = `
    <div class="result-verdict fade-up">
      <span class="verdict-icon">${icon}</span>
      <div class="verdict-label ${verdictCls}">${data.prediction}</div>
      <div class="verdict-confidence">${data.confidence.toFixed(1)}% confidence</div>
    </div>

    <div class="prob-bars fade-up fade-up-delay-1">
      <div class="prob-row">
        <div class="prob-header">
          <span class="text-green">Approval Probability</span>
          <span class="text-green">${data.approve_prob}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill prob-fill-green" style="width:0%" id="prob-green"></div>
        </div>
      </div>

      <div class="prob-row">
        <div class="prob-header">
          <span class="text-red">Rejection Probability</span>
          <span class="text-red">${data.reject_prob}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill prob-fill-red" style="width:0%" id="prob-red"></div>
        </div>
      </div>
    </div>

    <div class="risk-badge ${riskCls}" style="margin-top:16px;">
      ${riskIcon} ${data.risk_level}
    </div>

    <p style="font-size:12px;color:var(--text-muted);text-align:center;margin-top:16px;">
      Powered by XGBoost · 94.7% Training Accuracy
    </p>
  `;

  // Animate bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      const green = document.getElementById('prob-green');
      const red   = document.getElementById('prob-red');
      if (green) green.style.width = data.approve_prob + '%';
      if (red)   red.style.width   = data.reject_prob  + '%';
    }, 100);
  });
}

// ── Batch Upload ──────────────────────────────────────────────
const dropZone   = $('#dropZone');
const fileInput  = $('#csvFileInput');
const batchForm  = $('#batchForm');

if (dropZone && fileInput) {
  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      handleFileSelect(file);
    } else {
      showToast('Please upload a valid CSV file.', 'error');
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFileSelect(e.target.files[0]);
  });
}

function handleFileSelect(file) {
  const fileName = $('#selectedFileName');
  const uploadSection = $('#uploadSection');
  const processSection = $('#processSection');

  if (fileName)       fileName.textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  if (uploadSection)  uploadSection.style.display  = 'none';
  if (processSection) processSection.style.display = 'block';

  window._selectedBatchFile = file;
  showToast(`File selected: ${file.name}`, 'info');
}

const processBatchBtn = $('#processBatch');
if (processBatchBtn) {
  processBatchBtn.addEventListener('click', async () => {
    if (!window._selectedBatchFile) return;

    processBatchBtn.disabled = true;
    processBatchBtn.innerHTML = '<div class="spinner" style="width:18px;height:18px;border-width:2px;display:inline-block;"></div> Processing...';

    const progress = $('#progressFill');
    if (progress) {
      let p = 0;
      const iv = setInterval(() => {
        p = Math.min(p + Math.random() * 15, 90);
        progress.style.width = p + '%';
        if (p >= 90) clearInterval(iv);
      }, 200);
      window._progressInterval = iv;
    }

    const formData = new FormData();
    formData.append('file', window._selectedBatchFile);

    try {
      const response = await fetch('/api/batch', { method: 'POST', body: formData });
      const data     = await response.json();

      if (!response.ok || data.error) throw new Error(data.error || 'Batch processing failed');

      if (progress) {
        clearInterval(window._progressInterval);
        progress.style.width = '100%';
      }

      window._batchResults = data.results;
      renderBatchResults(data.results, data.summary);
      showToast(`Processed ${data.summary.total} applications successfully!`, 'success');

    } catch (err) {
      showToast('Batch error: ' + err.message, 'error');
    } finally {
      processBatchBtn.disabled = false;
      processBatchBtn.innerHTML = '🚀 Process Applications';
    }
  });
}

function renderBatchResults(results, summary) {
  const section = $('#resultsSection');
  if (!section) return;
  section.style.display = 'block';

  // Summary cards
  const summaryEl = $('#batchSummary');
  if (summaryEl) {
    summaryEl.innerHTML = `
      <div class="stat-card" style="--accent-gradient:linear-gradient(90deg,var(--indigo),var(--violet));">
        <div class="stat-icon" style="background:var(--violet-dim);">📋</div>
        <div class="stat-value">${summary.total}</div>
        <div class="stat-label">Total Applications</div>
      </div>
      <div class="stat-card" style="--accent-gradient:linear-gradient(90deg,#059669,var(--green));">
        <div class="stat-icon" style="background:var(--green-dim);">✅</div>
        <div class="stat-value text-green">${summary.approved}</div>
        <div class="stat-label">Approved</div>
      </div>
      <div class="stat-card" style="--accent-gradient:linear-gradient(90deg,#DC2626,var(--red));">
        <div class="stat-icon" style="background:var(--red-dim);">❌</div>
        <div class="stat-value text-red">${summary.rejected}</div>
        <div class="stat-label">Rejected</div>
      </div>
      <div class="stat-card" style="--accent-gradient:linear-gradient(90deg,var(--amber),#FBBF24);">
        <div class="stat-icon" style="background:var(--amber-dim);">📊</div>
        <div class="stat-value text-amber">${summary.approval_rate}%</div>
        <div class="stat-label">Approval Rate</div>
      </div>
    `;
  }

  // Results table
  const tbody = $('#resultsTableBody');
  if (tbody) {
    tbody.innerHTML = results.map(r => {
      const riskCls = r.risk_level === 'Low Risk' ? 'text-green' : r.risk_level === 'Moderate Risk' ? 'text-amber' : 'text-red';
      return `
        <tr>
          <td><code style="color:var(--violet-light);font-size:13px;">${r.loan_id}</code></td>
          <td>
            <span class="badge ${r.approved ? 'badge-approved' : 'badge-rejected'}">
              ${r.approved ? '✅' : '❌'} ${r.prediction}
            </span>
          </td>
          <td>
            <div style="display:flex;flex-direction:column;gap:3px;">
              <span>${r.approve_prob}%</span>
              <div class="progress-track" style="width:80px;">
                <div class="progress-fill" style="width:${r.approve_prob}%;background:${r.approved ? 'var(--green)' : 'var(--red)'};"></div>
              </div>
            </div>
          </td>
          <td class="${riskCls}" style="font-weight:600;font-size:13px;">${r.risk_level}</td>
        </tr>
      `;
    }).join('');
  }

  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Download button
const downloadBtn = $('#downloadResults');
if (downloadBtn) {
  downloadBtn.addEventListener('click', async () => {
    if (!window._batchResults) return;
    const response = await fetch('/api/batch/download', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ results: window._batchResults }),
    });
    const blob = await response.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'smart_lender_predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Results downloaded successfully!', 'success');
  });
}

// Reset batch upload
const resetBatchBtn = $('#resetBatch');
if (resetBatchBtn) {
  resetBatchBtn.addEventListener('click', () => {
    window._selectedBatchFile = null;
    window._batchResults = null;
    const upload  = $('#uploadSection');
    const process = $('#processSection');
    const results = $('#resultsSection');
    if (upload)  upload.style.display  = 'block';
    if (process) process.style.display = 'none';
    if (results) results.style.display = 'none';
    if (fileInput) fileInput.value     = '';
  });
}
