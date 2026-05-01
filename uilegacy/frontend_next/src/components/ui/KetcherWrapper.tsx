import { useEffect, useRef, useState } from 'react';
import { Ketcher, StandaloneStructServiceProvider } from 'ketcher-react';
import 'ketcher-react/dist/index.css';

interface KetcherWrapperProps {
  onMolExport: (smiles: string) => void;
  initialSmiles?: string;
}

const structServiceProvider = new StandaloneStructServiceProvider();

export const KetcherWrapper: React.FC<KetcherWrapperProps> = ({ onMolExport, initialSmiles }) => {
  const ketcherRef = useRef<any>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (initialSmiles && ketcherRef.current && loaded) {
      ketcherRef.current.setMolecule(initialSmiles).catch(console.warn);
    }
  }, [initialSmiles, loaded]);

  return (
    <Ketcher
      staticResourcesUrl="/static"
      structServiceProvider={structServiceProvider}
      onInit={(ketcher) => {
        ketcherRef.current = ketcher;
        setLoaded(true);
        ketcher.eventBus.on('action', async () => {
          try {
            const smiles = await ketcher.generateSmiles();
            onMolExport(smiles);
          } catch (e) {
            // ignore error during intermediate editing states
          }
        });
      }}
    />
  );
};
