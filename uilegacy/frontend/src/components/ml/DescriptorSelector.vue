<!-- frontend/src/components/ml/DescriptorSelector.vue -->
<template>
  <div class="descriptor-selector">
    <div class="header">
      <h3>🧬 Molecular Descriptors</h3>
      <el-button type="primary" size="small" plain @click="addAdapter" icon="Plus">Add Adapter</el-button>
    </div>

    <div v-if="selectedAdapters.length === 0" class="empty-state">
      <el-empty description="No descriptors selected. SMILES columns will be used as raw text." />
    </div>

    <el-collapse v-else v-model="activeNames" accordion>
      <el-collapse-item 
        v-for="(item, index) in selectedAdapters" 
        :key="index" 
        :name="index"
      >
        <template #title>
          <div class="collapse-title">
            <el-tag effect="dark">{{ item.key }}</el-tag>
            <span class="adapter-summary">{{ getSummary(item) }}</span>
            <el-button 
              type="danger" 
              size="small" 
              link 
              @click.stop="removeAdapter(index)" 
              icon="Delete"
            />
          </div>
        </template>

        <div class="adapter-config">
          <el-form-item label="Engine Type">
            <el-select v-model="item.key" @change="handleTypeChange(index)" class="w-full">
              <el-option 
                v-for="opt in availableAdapters" 
                :key="opt.key" 
                :label="opt.key" 
                :value="opt.key" 
              />
            </el-select>
          </el-form-item>

          <div v-if="loadingSchemas[index]" class="loading">
            <el-icon class="is-loading"><Loading /></el-icon> Loading parameters...
          </div>
          
          <ParamEditor 
            v-else-if="schemas[index]" 
            :schema="schemas[index]" 
            v-model="item.params" 
          />
        </div>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { getAdapters, getAdapterSchema } from '../../api/client'
import ParamEditor from './ParamEditor.vue'
import { Plus, Delete, Loading } from '@element-plus/icons-vue'

const props = defineProps<{
  modelValue: any[]
}>()

const emit = defineEmits(['update:modelValue'])

const availableAdapters = ref<any[]>([])
const selectedAdapters = ref<any[]>([])
const activeNames = ref([0])
const schemas = ref<Record<number, any>>({})
const loadingSchemas = ref<Record<number, boolean>>({})

onMounted(async () => {
  availableAdapters.value = await getAdapters()
  selectedAdapters.value = props.modelValue || []
  
  // Load initial schemas
  selectedAdapters.value.forEach((_, i) => loadSchema(i))
})

const addAdapter = () => {
  const defaultAdapter = availableAdapters.value[0]
  if (!defaultAdapter) return
  
  selectedAdapters.value.push({
    key: defaultAdapter.key,
    params: {}
  })
  
  const index = selectedAdapters.value.length - 1
  loadSchema(index)
  activeNames.value = [index]
  updateValue()
}

const removeAdapter = (index: number) => {
  selectedAdapters.value.splice(index, 1)
  updateValue()
}

const loadSchema = async (index: number) => {
  const adapter = selectedAdapters.value[index]
  if (!adapter) return
  
  loadingSchemas.value[index] = true
  try {
    const schema = await getAdapterSchema(adapter.key)
    schemas.value[index] = schema
    
    // Initialize params with defaults from schema if empty
    schema.forEach((s: any) => {
      if (adapter.params[s.name] === undefined) {
        adapter.params[s.name] = s.default
      }
    })
  } catch (e) {
    console.error('Failed to load adapter schema', e)
  } finally {
    loadingSchemas.value[index] = false
  }
}

const handleTypeChange = (index: number) => {
  selectedAdapters.value[index].params = {}
  loadSchema(index)
  updateValue()
}

const getSummary = (item: any) => {
  const p = item.params
  if (item.key === 'RDKit') return `Size: ${p.fpSize || 2048}, Radius: ${p.radius || 2}`
  if (item.key === 'Mordred') return `Include 3D: ${p.ignore_3d ? 'No' : 'Yes'}`
  return ''
}

const updateValue = () => {
  emit('update:modelValue', selectedAdapters.value)
}

watch(selectedAdapters, updateValue, { deep: true })
</script>

<style scoped>
.descriptor-selector { margin-bottom: 20px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
.header h3 { margin: 0; font-size: 1rem; color: #409eff; }
.empty-state { padding: 20px; background: rgba(255, 255, 255, 0.02); border-radius: 8px; border: 1px dashed #dcdfe6; }
.collapse-title { display: flex; align-items: center; gap: 12px; width: 100%; padding-right: 15px; }
.adapter-summary { flex-grow: 1; font-size: 0.8rem; color: #909399; }
.adapter-config { padding: 10px 0; }
.loading { padding: 10px; text-align: center; color: #909399; font-size: 0.85rem; }
.w-full { width: 100%; }
</style>
