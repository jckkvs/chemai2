<!-- frontend/src/components/ml/MonotonicConstraints.vue -->
<template>
  <div class="monotonic-constraints">
    <div class="header">
      <h3>📈 Monotonic Constraints</h3>
      <el-tooltip content="Force the model output to either increase or decrease relative to specific features." placement="top">
        <el-icon class="info-icon"><InfoFilled /></el-icon>
      </el-tooltip>
    </div>

    <div v-if="availableColumns.length === 0" class="empty-state">
      <el-empty description="Upload data to configure constraints" :image-size="60" />
    </div>

    <div v-else class="constraints-table-wrapper">
      <el-table :data="constraintList" size="small" border stripe>
        <el-table-column label="Feature Column" prop="col" min-width="180">
          <template #default="{ row }">
            <span class="col-name">{{ row.col }}</span>
            <el-tag v-if="row.col === store.targetCol" size="small" type="info" class="ml-2">Target</el-tag>
          </template>
        </el-table-column>
        
        <el-table-column label="Constraint" width="220">
          <template #default="{ row }">
            <el-radio-group v-model="row.direction" size="small" @change="updateValue">
              <el-radio-button :label="0">None</el-radio-button>
              <el-radio-button :label="1"><el-icon><TopRight /></el-icon> Inc</el-radio-button>
              <el-radio-button :label="-1"><el-icon><BottomRight /></el-icon> Dec</el-radio-button>
            </el-radio-group>
          </template>
        </el-table-column>
      </el-table>
    </div>
    
    <p v-if="activeCount > 0" class="active-hint">
      <el-icon><Check /></el-icon> {{ activeCount }} active constraints applied
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useChemaiStore } from '../../store/chemai'
import { InfoFilled, TopRight, BottomRight, Check } from '@element-plus/icons-vue'

const props = defineProps<{
  modelValue: Record<string, number>
}>()

const emit = defineEmits(['update:modelValue'])
const store = useChemaiStore()

const constraintList = ref<any[]>([])

const availableColumns = computed(() => {
  return store.columns.filter(c => c !== store.targetCol)
})

const activeCount = computed(() => {
  return constraintList.value.filter(c => c.direction !== 0).length
})

onMounted(() => {
  syncFromValue()
})

const syncFromValue = () => {
  constraintList.value = availableColumns.value.map(col => ({
    col,
    direction: props.modelValue[col] || 0
  }))
}

// Re-sync when columns change (e.g. new file upload)
watch(() => store.columns, syncFromValue)

const updateValue = () => {
  const result: Record<string, number> = {}
  constraintList.value.forEach(item => {
    if (item.direction !== 0) {
      result[item.col] = item.direction
    }
  })
  emit('update:modelValue', result)
}
</script>

<style scoped>
.monotonic-constraints { margin-bottom: 20px; }
.header { display: flex; align-items: center; gap: 8px; margin-bottom: 15px; }
.header h3 { margin: 0; font-size: 1rem; color: #409eff; }
.info-icon { color: #909399; cursor: help; }
.empty-state { padding: 20px; background: rgba(255, 255, 255, 0.02); border-radius: 8px; }
.constraints-table-wrapper { max-height: 400px; overflow-y: auto; border-radius: 8px; border: 1px solid #ebeef5; }
.col-name { font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
.active-hint { margin-top: 10px; font-size: 0.8rem; color: #67c23a; display: flex; align-items: center; gap: 4px; font-weight: bold; }
</style>
