const DEFAULT_MATERIAL = 'SiC';
const DEBOUNCE_MS = 320;

const state = {
  material: DEFAULT_MATERIAL,
  config: null,
  controls: {},
  debounceTimer: null,
  isPredicting: false
};

const elements = {
  materialSelect: document.getElementById('materialSelect'),
  materialBadge: document.getElementById('materialBadge'),
  controlPanel: document.getElementById('controlPanel'),
  recipesGrid: document.getElementById('recipesGrid'),
  metricsGrid: document.getElementById('metricsGrid'),
  featurePreview: document.getElementById('featurePreview'),
  diagnosisText: document.getElementById('diagnosisText'),
  lastUpdated: document.getElementById('lastUpdated'),
  refreshButton: document.getElementById('refreshButton'),
  statusDot: document.getElementById('statusDot'),
  statusTitle: document.getElementById('statusTitle'),
  statusCopy: document.getElementById('statusCopy')
};

const apiClient = window.api || {
  getMaterialConfig: (materialName) => fetchJson(`/api/config/${encodeURIComponent(materialName)}`),
  predictMaterial: (materialName, features) => fetchJson(`/api/predict/${encodeURIComponent(materialName)}`, {
    method: 'POST',
    body: JSON.stringify({ features })
  }),
  getRecipes: (materialName) => fetchJson(`/api/recipes/${encodeURIComponent(materialName)}`),
  healthCheck: () => fetchJson('/api/health')
};

document.addEventListener('DOMContentLoaded', () => {
  elements.materialSelect.value = DEFAULT_MATERIAL;
  elements.materialSelect.addEventListener('change', handleMaterialChange);
  elements.refreshButton.addEventListener('click', () => loadMaterial(state.material));
  loadMaterial(DEFAULT_MATERIAL);
});

async function fetchJson(path, options = {}) {
  const response = await fetch(`http://127.0.0.1:8000${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `API request failed: ${response.status}`);
  }
  return payload;
}

async function handleMaterialChange(event) {
  const nextMaterial = event.target.value;

  // 切換材料時要完整重置本地控制狀態。
  // 原因是 SiC 與 NiO 的 search_space 不同，例如 NiO 有 Current_Compliance_A，
  // 若沿用前一個材料的 slider value，會把不存在或超出範圍的特徵送進模型。
  state.controls = {};
  state.config = null;
  state.material = nextMaterial;
  await loadMaterial(nextMaterial);
}

async function loadMaterial(materialName) {
  state.material = materialName;
  setStatus('loading', `正在載入 ${materialName}`, '讀取材料設定與 Pareto 推薦處方。');
  renderLoading();

  try {
    const [config, recipes] = await Promise.all([
      apiClient.getMaterialConfig(materialName),
      apiClient.getRecipes(materialName).catch(() => [])
    ]);

    state.config = config;
    elements.materialBadge.textContent = config.material;
    renderControls(config);
    renderRecipes(recipes);
    updateFeaturePreview();
    await predictCurrentRecipe();
    setStatus('ready', `${config.material} 已就緒`, '後端模型已連線，調整參數會即時更新。');
  } catch (error) {
    setStatus('error', '後端連線失敗', error.message);
    renderError(error.message);
  }
}

function renderLoading() {
  elements.controlPanel.innerHTML = `
    <div class="skeleton-line"></div>
    <div class="skeleton-line short"></div>
    <div class="skeleton-line"></div>
  `;
  elements.recipesGrid.innerHTML = '<article class="recipe-card loading-card">載入推薦處方中...</article>';
  elements.metricsGrid.innerHTML = '<article class="metric-card loading-card">等待模型預測...</article>';
}

function renderControls(config) {
  const searchSpace = config.search_space || {};
  const controlFeatures = Object.keys(searchSpace);

  elements.controlPanel.innerHTML = '';

  controlFeatures.forEach((featureName) => {
    const spec = searchSpace[featureName] || {};
    const group = document.createElement('div');
    group.className = 'control-group';
    group.dataset.feature = featureName;

    const labelRow = document.createElement('div');
    labelRow.className = 'control-topline';
    labelRow.innerHTML = `
      <span class="control-name">${formatFeatureName(featureName)}</span>
      <span class="control-value" id="value-${cssSafe(featureName)}"></span>
    `;

    const input = createControlInput(featureName, spec, config);

    group.appendChild(labelRow);
    group.appendChild(input);
    elements.controlPanel.appendChild(group);
    updateControlValueLabel(featureName);
  });

  if (controlFeatures.length === 0) {
    elements.controlPanel.innerHTML = '<p class="empty-state">此材料尚未定義可調參數。</p>';
  }
}

function createControlInput(featureName, spec, config) {
  const choices = Array.isArray(spec.choices) ? spec.choices : null;
  const isCategorical = spec.param_type === 'categorical' || Boolean(choices);

  if (isCategorical) {
    const select = document.createElement('select');
    select.className = 'control-select';
    select.dataset.feature = featureName;

    const values = choices && choices.length > 0
      ? choices
      : [config.default_feature_values?.[featureName] ?? 0];

    values.forEach((choice) => {
      const option = document.createElement('option');
      option.value = Number(choice);
      option.textContent = formatFeatureValue(featureName, Number(choice));
      select.appendChild(option);
    });

    const defaultValue = state.controls[featureName] ?? Number(values[0]);
    select.value = String(nearestChoice(defaultValue, values));
    state.controls[featureName] = Number(select.value);

    select.addEventListener('change', (event) => {
      state.controls[featureName] = Number(event.target.value);
      updateControlValueLabel(featureName);
      schedulePrediction();
    });
    return select;
  }

  const range = document.createElement('input');
  range.type = 'range';
  range.dataset.feature = featureName;
  range.min = Number(spec.low ?? 0);
  range.max = Number(spec.high ?? 100);
  range.step = Number(spec.step ?? 1);

  const defaultValue = state.controls[featureName] ?? midpoint(Number(range.min), Number(range.max));
  range.value = clamp(defaultValue, Number(range.min), Number(range.max));
  state.controls[featureName] = Number(range.value);

  range.addEventListener('input', (event) => {
    state.controls[featureName] = Number(event.target.value);
    updateControlValueLabel(featureName);
    updateFeaturePreview();
    schedulePrediction();
  });
  return range;
}

function schedulePrediction() {
  window.clearTimeout(state.debounceTimer);
  state.debounceTimer = window.setTimeout(() => {
    predictCurrentRecipe();
  }, DEBOUNCE_MS);
}

async function predictCurrentRecipe() {
  if (!state.config || state.isPredicting) {
    return;
  }

  const features = buildFeaturePayload();
  updateFeaturePreview(features);
  state.isPredicting = true;
  setStatus('loading', '模型推論中', '正在計算電性平均值與 95% 信心區間。');

  try {
    const response = await apiClient.predictMaterial(state.material, features);
    renderPredictions(response.predictions || {});
    renderDiagnosis(features, response.predictions || {});
    elements.lastUpdated.textContent = `最後更新 ${new Date().toLocaleTimeString('zh-TW', { hour12: false })}`;
    setStatus('ready', `${state.material} 預測完成`, 'GPR 預測與信心區間已更新。');
  } catch (error) {
    elements.metricsGrid.innerHTML = `<article class="metric-card loading-card">預測失敗：${escapeHtml(error.message)}</article>`;
    setStatus('error', '預測失敗', error.message);
  } finally {
    state.isPredicting = false;
  }
}

function buildFeaturePayload() {
  const config = state.config || {};
  const features = { ...state.controls };

  Object.entries(config.default_feature_values || {}).forEach(([feature, value]) => {
    if (features[feature] === undefined || Number.isNaN(Number(features[feature]))) {
      features[feature] = Number(value);
    }
  });

  if ((config.feature_columns || []).includes('Has_RTA') && features.RTA_Temperature_C !== undefined) {
    features.Has_RTA = Number(features.RTA_Temperature_C) === Number(config.no_rta_temperature_c) ? 0 : 1;
  }

  return features;
}

function renderPredictions(predictions) {
  const entries = Object.entries(predictions);
  if (entries.length === 0) {
    elements.metricsGrid.innerHTML = '<article class="metric-card loading-card">目前沒有可用模型輸出。</article>';
    return;
  }

  const preferredOrder = [
    'Leakage_Current_A',
    'Endurance_Cycles',
    'On_Off_Ratio',
    'Operation_Voltage_V',
    'Forming_Voltage_V'
  ];

  entries.sort(([left], [right]) => {
    const leftIndex = preferredOrder.indexOf(left);
    const rightIndex = preferredOrder.indexOf(right);
    return (leftIndex === -1 ? 99 : leftIndex) - (rightIndex === -1 ? 99 : rightIndex);
  });

  elements.metricsGrid.innerHTML = entries.map(([target, metric]) => `
    <article class="metric-card">
      <p class="metric-label">${formatTargetName(target)}</p>
      <p class="metric-value">${formatMetricValue(target, metric.mean)}</p>
      <p class="metric-ci">95% CI：${formatMetricValue(target, metric.ci95_low)} - ${formatMetricValue(target, metric.ci95_high)}</p>
    </article>
  `).join('');
}

function renderRecipes(recipes) {
  if (!Array.isArray(recipes) || recipes.length === 0) {
    elements.recipesGrid.innerHTML = '<article class="recipe-card loading-card">尚未找到 Pareto 推薦處方。</article>';
    return;
  }

  elements.recipesGrid.innerHTML = '';
  recipes.slice(0, 3).forEach((recipe) => {
    const card = document.createElement('article');
    card.className = 'recipe-card';
    card.innerHTML = `
      <div>
        <p class="eyebrow">${formatRecipeLabel(recipe.label)}</p>
        <h4 class="recipe-title">${recipe.secondary_target || 'Sweet Spot'}</h4>
      </div>
      <p class="recipe-strategy">${escapeHtml(recipe.strategy || 'AI 推薦的可行製程區間')}</p>
      <ul class="recipe-feature-list">
        ${Object.entries(recipe.features || {}).map(([feature, value]) => `
          <li><span>${formatFeatureName(feature)}</span><strong>${formatFeatureValue(feature, value)}</strong></li>
        `).join('')}
      </ul>
      <button class="apply-button" type="button">套用此參數</button>
    `;

    card.querySelector('button').addEventListener('click', () => applyRecipe(recipe.features || {}));
    elements.recipesGrid.appendChild(card);
  });
}

function applyRecipe(features) {
  const searchSpace = state.config?.search_space || {};

  Object.keys(searchSpace).forEach((featureName) => {
    if (features[featureName] === undefined) {
      return;
    }
    const value = Number(features[featureName]);
    state.controls[featureName] = value;

    const input = elements.controlPanel.querySelector(`[data-feature="${featureName}"]`);
    if (!input) {
      return;
    }

    if (input.tagName === 'SELECT') {
      const options = Array.from(input.options).map((option) => Number(option.value));
      input.value = String(nearestChoice(value, options));
      state.controls[featureName] = Number(input.value);
    } else {
      input.value = String(clamp(value, Number(input.min), Number(input.max)));
      state.controls[featureName] = Number(input.value);
    }
    updateControlValueLabel(featureName);
  });

  updateFeaturePreview();
  predictCurrentRecipe();
}

function renderDiagnosis(features, predictions) {
  const leakage = predictions.Leakage_Current_A?.mean;
  const endurance = predictions.Endurance_Cycles?.mean;
  const ratio = predictions.On_Off_Ratio?.mean;
  const voltage = predictions.Operation_Voltage_V?.mean;

  const statements = [];
  if (Number.isFinite(leakage)) {
    statements.push(`漏電流預測約為 ${formatMetricValue('Leakage_Current_A', leakage)}，${leakage < 1e-6 ? '屬於偏低漏電區間' : '仍需留意缺陷輔助導通'}。`);
  }
  if (Number.isFinite(ratio)) {
    statements.push(`On/Off ratio 約為 ${formatMetricValue('On_Off_Ratio', ratio)}，${ratio >= 5 ? '滿足可用記憶窗門檻' : '記憶窗仍偏窄'}。`);
  }
  if (Number.isFinite(endurance)) {
    statements.push(`Endurance 預測為 ${formatMetricValue('Endurance_Cycles', endurance)}，可作為下一輪循環量測優先條件。`);
  }
  if (Number.isFinite(voltage)) {
    statements.push(`操作電壓約 ${formatMetricValue('Operation_Voltage_V', voltage)}，可用於評估低功耗操作可行性。`);
  }

  elements.diagnosisText.textContent = statements.length > 0
    ? statements.join(' ')
    : `目前 ${state.material} 參數已送出，但模型尚未回傳可解讀的核心電性指標。`;
}

function updateFeaturePreview(features = buildFeaturePayload()) {
  elements.featurePreview.textContent = JSON.stringify(features, null, 2);
}

function updateControlValueLabel(featureName) {
  const label = document.getElementById(`value-${cssSafe(featureName)}`);
  if (!label) {
    return;
  }
  label.textContent = formatFeatureValue(featureName, state.controls[featureName]);
}

function setStatus(mode, title, copy) {
  elements.statusDot.classList.remove('ready', 'error');
  if (mode === 'ready') {
    elements.statusDot.classList.add('ready');
  }
  if (mode === 'error') {
    elements.statusDot.classList.add('error');
  }
  elements.statusTitle.textContent = title;
  elements.statusCopy.textContent = copy;
}

function renderError(message) {
  const escaped = escapeHtml(message);
  elements.controlPanel.innerHTML = `<p class="empty-state">${escaped}</p>`;
  elements.recipesGrid.innerHTML = `<article class="recipe-card loading-card">${escaped}</article>`;
  elements.metricsGrid.innerHTML = `<article class="metric-card loading-card">${escaped}</article>`;
}

function formatFeatureName(name) {
  const labels = {
    RF_Power_W: 'RF Power',
    Process_Time_Min: 'Process Time',
    RTA_Temperature_C: 'RTA Temperature',
    Current_Compliance_A: 'Current Compliance',
    Has_RTA: 'Has RTA'
  };
  return labels[name] || name.replaceAll('_', ' ');
}

function formatTargetName(name) {
  const labels = {
    Leakage_Current_A: 'Leakage Current',
    Endurance_Cycles: 'Endurance',
    On_Off_Ratio: 'On/Off Ratio',
    Operation_Voltage_V: 'Operation Voltage',
    Forming_Voltage_V: 'Forming Voltage'
  };
  return labels[name] || name.replaceAll('_', ' ');
}

function formatFeatureValue(featureName, value) {
  const numeric = Number(value);
  if (featureName === 'RTA_Temperature_C') {
    return numeric === 25 ? 'No RTA' : `${numeric.toFixed(0)} °C`;
  }
  if (featureName === 'RF_Power_W') {
    return `${numeric.toFixed(0)} W`;
  }
  if (featureName === 'Process_Time_Min') {
    return `${numeric.toFixed(0)} min`;
  }
  if (featureName === 'Current_Compliance_A') {
    return `${numeric.toExponential(1)} A`;
  }
  return Number.isInteger(numeric) ? numeric.toFixed(0) : numeric.toPrecision(4);
}

function formatMetricValue(target, value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 'N/A';
  }
  if (target === 'Leakage_Current_A') {
    return `${numeric.toExponential(2)} A`;
  }
  if (target.includes('Voltage')) {
    return `${numeric.toFixed(2)} V`;
  }
  if (target === 'Endurance_Cycles') {
    return `${Math.round(numeric).toLocaleString('zh-TW')} cycles`;
  }
  if (target === 'On_Off_Ratio') {
    return numeric >= 100 ? numeric.toExponential(2) : numeric.toFixed(2);
  }
  return numeric.toPrecision(4);
}

function formatRecipeLabel(label) {
  const labels = {
    ultra_low_leakage: 'Ultra-Low Leakage',
    high_secondary_objective: 'High Stability',
    balanced_sweet_spot: 'Balanced Sweet Spot'
  };
  return labels[label] || String(label || 'Recommended Recipe').replaceAll('_', ' ');
}

function nearestChoice(value, choices) {
  return choices.reduce((best, current) => {
    return Math.abs(Number(current) - Number(value)) < Math.abs(Number(best) - Number(value))
      ? current
      : best;
  }, choices[0]);
}

function clamp(value, min, max) {
  return Math.min(Math.max(Number(value), min), max);
}

function midpoint(min, max) {
  return (min + max) / 2;
}

function cssSafe(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]/g, '-');
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
