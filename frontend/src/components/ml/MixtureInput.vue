<!-- frontend/src/components/ml/MixtureInput.vue -->
<template>
  <div class="mixture-input">
    <div class="header">
      <div class="title-with-icon">
        <el-icon class="main-icon"><Blender /></el-icon>
        <div>
          <h3>Mixture Formulation</h3>
          <p class="subtitle">Define multi-component mixtures and calculate weighted descriptors</p>
        </div>
      </div>
      <el-button type="primary" size="small" plain @click="addComponent" icon="Plus">Add Component</el-button>
    </div>

    <el-card class="form-card" shadow="never">
      <div class="ratio-type-selector">
        <span class="label">Ratio Unit:</span>
        <el-radio-group v-model="ratioType" size="small">
          <el-radio-button label="weight">Weight Ratio</el-radio-button>
          <el-radio-button label="mole">Mole Ratio</el-radio-button>
          <el-radio-button label="other">Other</el-radio-button>
        </el-radio-group>
        <el-input 
          v-if="ratioType === 'other'" 
          v-model="otherUnit" 
          placeholder="e.g. volume_fraction" 
          size="small" 
          class="ml-2 w-32" 
        />
      </div>

      <div class="components-list">
        <div v-for="(comp, index) in components" :key="index" class="component-row animate-fade-in">
          <div class="index">#{{ index + 1 }}</div>
          <el-input v-model="comp.smiles" placeholder="SMILES (e.g. CCO)" class="smiles-input" />
          <el-input v-model="comp.compound_name" placeholder="Name (optional)" class="name-input" />
          <el-input-number v-model="comp.ratio_value" :min="0.001" :step="0.1" class="ratio-input" controls-position="right" />
          <el-button type="danger" icon="Delete" circle size="small" @click="removeComponent(index)" :disabled="components.length <= 2" />
        </div>
      </div>

      <div class="actions">
        <el-button 
          type="primary" 
          class="run-btn" 
          :loading="loading" 
          @click="calculateMixture"
          icon="MagicStick"
        >
          Calculate Weighted Descriptors
        </el-button>
      </div>
    </el-card>

    <!-- Results Panel -->
    <div v-if="result" class="results-panel animate-fade-in">
      <el-alert 
        title="Calculation Successful" 
        type="success" 
        :description="`Generated ${Object.keys(result.mixture_features).length} weighted features.`" 
        show-icon 
        class="mb-4"
      />
      
      <el-table :data="tableData" border stripe size="small">
        <el-table-column label="#" width="50" prop="index" />
        <el-table-column label="SMILES" prop="smiles" min-width="150" />
        <el-table-column label="MW" prop="mw" width="80" />
        <el-table-column label="Weight %" prop="weightPct" width="100" />
        <el-table-column label="Mole %" prop="molePct" width="100" />
      </el-table>

      <div v-if="result.warnings.length > 0" class="warnings mt-4">
        <div v-for="w in result.warnings" :key="w" class="text-amber-500 text-xs flex items-center gap-1">
          <el-icon><Warning /></el-icon> {{ w }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { api } from '../../api/client'
import { ElMessage } from 'element-plus'
import { Blender, Plus, Delete, MagicStick, Warning } from '@element-plus/icons-vue'

const ratioType = ref('weight')
const otherUnit = ref('')
const loading = ref(false)
const result = ref<any>(null)

const components = ref([
  { smiles: '', compound_name: '', ratio_value: 1.0 },
  { smiles: '', compound_name: '', ratio_value: 1.0 }
])

const addComponent = () => {
  components.value.push({ smiles: '', compound_name: '', ratio_value: 1.0 })
}

const removeComponent = (index: number) => {
  components.value.splice(index, 1)
}

const calculateMixture = async () => {
  if (components.value.some(c => !c.smiles.trim())) {
    ElMessage.error('Please provide SMILES for all components')
    return
  }

  loading.value = true
  try {
    const res = await api.post('/mixture/calculate', {
      components: components.value.map(c => ({
        ...c,
        ratio_unit: ratioType.value === 'other' ? otherUnit.value : ratioType.value
      }))
    })
    result.value = res.data
    ElMessage.success('Mixture features calculated')
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || 'Calculation failed')
  } finally {
    loading.value = false
  }
}

const tableData = computed(() => {
  if (!result.value) return []
  const info = result.value.conversion_info
  return components.value.map((c, i) => ({
    index: i + 1,
    smiles: c.smiles,
    mw: info.molecular_weights[i].toFixed(2),
    weightPct: (info.weight_fractions[i] * 100).toFixed(1) + '%',
    molePct: (info.mole_fractions[i] * 100).toFixed(1) + '%'
  }))
})
</script>

<style scoped>
.mixture-input { margin-bottom: 20px; }
.header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 20px; }
.title-with-icon { display: flex; gap: 12px; }
.main-icon { font-size: 2rem; color: #a78bfa; margin-top: 4px; }
.header h3 { margin: 0; font-size: 1.1rem; color: #e0e0f0; }
.subtitle { margin: 2px 0 0; font-size: 0.85rem; color: #909399; }

.form-card { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; }
.ratio-type-selector { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
.label { font-size: 0.9rem; font-weight: 600; color: #606266; }

.components-list { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
.component-row { display: flex; align-items: center; gap: 10px; padding: 10px; background: rgba(255, 255, 255, 0.03); border-radius: 8px; }
.index { width: 30px; font-weight: bold; color: #409eff; text-align: center; }
.smiles-input { flex-grow: 1; }
.name-input { width: 150px; }
.ratio-input { width: 120px; }

.actions { display: flex; justify-content: center; }
.run-btn { width: 100%; height: 44px; font-size: 1rem; border-radius: 10px; background: linear-gradient(135deg, #7b2ff7, #00d4ff) !important; border: none; font-weight: bold; }

.results-panel { margin-top: 25px; padding: 20px; background: rgba(74, 222, 128, 0.05); border-radius: 12px; border: 1px solid rgba(74, 222, 128, 0.1); }
.w-32 { width: 8rem; }
.mb-4 { margin-bottom: 1rem; }
.mt-4 { margin-top: 1rem; }
.ml-2 { margin-left: 0.5rem; }
</style>
